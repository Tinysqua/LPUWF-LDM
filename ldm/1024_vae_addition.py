import os
import random
import sys; sys.path.append('./')
from utils.Condition_dataloader import split_dataloader, double_form_dataloader
from utils.common import load_config, get_parameters, copy_yaml_to_folder
from diffusers import AutoencoderKL
from ldm.contrl_autoencoder import AutoencoderKL_output, AutoencoderKL_input
from generative.losses import PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from generative.losses.adversarial_loss import PatchAdversarialLoss
import argparse
import torch
from accelerate import Accelerator
from tqdm import tqdm
from torch.nn import functional as F
from torchvision.utils import make_grid
from os.path import join as j
from accelerate.utils import DistributedDataParallelKwargs

def main(args):
    cf = load_config(args.config_path)
    kwargs = DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=True)
    accelerator = Accelerator(**get_parameters(Accelerator, cf), kwargs_handlers=[kwargs])
    if accelerator.is_main_process:
        copy_yaml_to_folder(args.config_path, cf['project_dir'])
    train_dataloader = double_form_dataloader(cf['data_path'], 
                                              cf['sample_size'], 
                                              cf['train_bc'], 
                                              'double', 
                                              data_aug=cf['data_aug'])
    val_dataloader = double_form_dataloader(cf['eval_path'], 
                                            cf['sample_size'], 
                                            cf['eval_bc'], 
                                            'double', 
                                            data_aug=cf['data_aug'])

    # old_autoencoder_in = AutoencoderKL.from_pretrained(cf['resume_path'])
    # old_autoencoder_out = AutoencoderKL.from_pretrained(cf['resume_path_con'])
    output_autoencoder = AutoencoderKL_output.from_pretrained(cf['resume_path_con'])
    input_autoencoder = AutoencoderKL_input.from_pretrained(cf['resume_path'])
    # input_autoencoder.to(accelerator.device)
    # input_autoencoder.requires_grad_(False)
    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
    perceptual_loss.to(accelerator.device)
    perceptual_weight = 0.001

    discriminator = PatchDiscriminator(spatial_dims=2, 
                                num_layers_d=3, num_channels=64, in_channels=3, out_channels=1)
    discriminator.load_state_dict(torch.load(cf['dis_path']))
    
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01
    optimizer_g = torch.optim.Adam(list(output_autoencoder.parameters())+list(input_autoencoder.parameters()), 
                                   lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)
    
    # future in config
    kl_weight = 1e-6
    autoencoder_warm_up_n_epochs = 0
    val_interval = cf['val_inter']
    save_interval = cf['save_inter']

    if len(cf['log_with']):
        accelerator.init_trackers('train_example')
    output_autoencoder, input_autoencoder, discriminator, optimizer_g, optimizer_d, train_dataloader, val_dataloader = accelerator.prepare(
        output_autoencoder, input_autoencoder, discriminator, optimizer_g, optimizer_d, train_dataloader, val_dataloader
    )
    
    global_step = 0
    for epoch in range(cf['num_epochs']):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}")
        for step, batch in enumerate(train_dataloader):
            batch = batch[0]
            index = random.choice([1, 2])
            images, conditions = batch[index], batch[0]
        
            recon_output, posterior_output, res = output_autoencoder(conditions, return_dict=False, sample_posterior=False)
            recon, posterior_input = input_autoencoder(images, add_res=res, sample_posterior=False, return_dict=False)
            recon_loss = F.mse_loss(recon, images)
            p_loss = perceptual_loss(recon, images)
            kl_loss = posterior_output.kl().mean() + posterior_input.kl().mean()
            loss_g = recon_loss + (perceptual_weight * p_loss) + (kl_weight * kl_loss)
            
            if epoch+1 > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(recon.contiguous())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss
            
        
            accelerator.backward(loss_g)
            # print('Grad: ', list(input_autoencoder.parameters())[0].grad)
            optimizer_g.step()
            optimizer_g.zero_grad()
            
            if epoch+1 > autoencoder_warm_up_n_epochs:
                with accelerator.accumulate(discriminator):
                    with torch.no_grad():
                        _, _, res = output_autoencoder(conditions, return_dict=False, sample_posterior=False)
                        recon = input_autoencoder(images, add_res=res, 
                                                  sample_posterior=False).sample
                    logits_fake = discriminator(recon.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                    
                    loss_d = adv_weight * discriminator_loss
                    accelerator.backward(loss_d)
                    optimizer_d.step()
                    optimizer_d.zero_grad()
                    
            progress_bar.update(1)
            logs = {"loss": recon_loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            val_model_output = accelerator.unwrap_model(output_autoencoder)
            val_model_input = accelerator.unwrap_model(input_autoencoder)

            if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
                with torch.no_grad():
                    batch = next(iter(val_dataloader))[0]
                    conditions, images = batch[0], batch[1]
                    # batch = next(iter(val_dataloader))
                    # conditions, images = batch[1], batch[0]
                    _, _, res = val_model_output(conditions, return_dict=False, sample_posterior=False)
                    val_recon = val_model_input(images, add_res=res, sample_posterior=False).sample
                    val_recon = torch.cat([conditions, images, val_recon], dim=-1)
                    val_recon = make_grid(val_recon, nrow=1).unsqueeze(0)
                    log_image = {"Eval": (val_recon+1)/2}
                    accelerator.trackers[0].log_images(log_image, epoch+1)
                
            if (epoch + 1) % save_interval == 0 or epoch == cf['num_epochs'] - 1:
                val_model_output.save_pretrained(j(cf['project_dir'], 'output_save'))
                val_model_input.save_pretrained(j(cf['project_dir'], 'input_save'))
                dis_save = j(cf['project_dir'], 'dis_save')
                os.makedirs(dis_save, exist_ok=True)
                torch.save(accelerator.unwrap_model(discriminator).state_dict(), j(dis_save, 'discriminator.pt'))
                
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/1024_vae_addition_test_config.yaml')

    args = parser.parse_args()
    main(args)