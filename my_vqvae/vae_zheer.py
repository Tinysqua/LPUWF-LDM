import os
import random
import sys; sys.path.append('./')
from utils.Condition_dataloader import split_dataloader, double_form_dataloader
from utils.common import save_config_to_yaml, get_parameters, copy_yaml_to_folder
from diffusers import AutoencoderKL
from generative.losses import PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from generative.losses.adversarial_loss import PatchAdversarialLoss
import torch
from config.vae.config_vae_zheer import Config
from accelerate import Accelerator
from tqdm import tqdm
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from os.path import join as j
from accelerate.utils import DistributedDataParallelKwargs

def main():
    cf = save_config_to_yaml(Config, Config.project_dir)
    kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
    accelerator = Accelerator(**get_parameters(Accelerator, cf), kwargs_handlers=[kwargs])
    train_dataloader = double_form_dataloader(Config.data_path, 
                                        Config.sample_size, 
                                        Config.train_bc, 
                                        Config.mode)
    device = 'cuda'
    val_dataloader = double_form_dataloader(
        Config.eval_path, 
        Config.sample_size, 
        Config.eval_bc, 
        Config.mode)
    
    if len(Config.resume_path):
        print("I'm here!")
        autoencoderkl = AutoencoderKL.from_pretrained(Config.resume_path)
    else:
        autoencoderkl = AutoencoderKL(
            in_channels=cf['in_channels'], 
            out_channels=cf['out_channels'], 
            down_block_types=cf['down_block_types'], 
            up_block_types=cf['up_block_types'], 
            latent_channels=cf['latent_channels'], 
            layers_per_block=cf['layers_per_block'], 
            sample_size=cf['sample_size'], 
            block_out_channels=cf['block_out_channels']
        )
    autoencoderkl = autoencoderkl.to(device)
    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
    perceptual_loss.to(device)
    perceptual_weight = 0.001

    discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, 
                                       num_channels=64, in_channels=3, out_channels=1).to(device)
    
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01
    optimizer_g = torch.optim.Adam(autoencoderkl.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)
    
    # future in config
    kl_weight = 1e-6
    autoencoder_warm_up_n_epochs = 0
    val_interval = Config.val_inter
    save_interval = Config.save_inter

    if len(Config.log_with):
        accelerator.init_trackers('train_example')    
    
    global_step = 0
    for epoch in range(Config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch+1}")
        autoencoderkl.train()
        discriminator.train()
        for step, batch in enumerate(train_dataloader):
            batch = batch[0]
            images = batch[random.randint(1,2)].to(device)

            print("Image size: ", images.shape)
            optimizer_g.zero_grad(set_to_none=True)
            recon, posterior = autoencoderkl(images, sample_posterior=True, return_dict=False)
            recon_loss = F.l1_loss(recon, images)
            p_loss = perceptual_loss(recon, images)
            kl_loss = posterior.kl().mean()
            loss_g = recon_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)
            
            if epoch+1 > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(recon.contiguous())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss
            
            loss_g.backward()
            optimizer_g.step()
                
            if epoch+1 > autoencoder_warm_up_n_epochs:
                optimizer_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    recon, posterior = autoencoderkl(images, 
                                        sample_posterior=True, return_dict=False)
                logits_fake = discriminator(recon.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                
                loss_d = adv_weight * discriminator_loss
                loss_d.backward()
                optimizer_d.step()
                    
            progress_bar.update(1)
            logs = {"gen_loss": loss_g.detach().item(), 
                    "dis_loss": loss_d.detach().item() if epoch+1 > Config.autoencoder_warm_up_n_epochs else 0}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            
        if accelerator.is_main_process:
            val_model = autoencoderkl.eval()
            if (epoch + 1) % val_interval == 0 or epoch == Config.num_epochs - 1:
                with torch.no_grad():
                    save_path = j(Config.project_dir, 'image_save')
                    os.makedirs(save_path, exist_ok=True)
                    batch = next(iter(val_dataloader))[0]
                    first_ffa = batch[1].to(device)
                    second_ffa = batch[2].to(device)
                    val_recon = val_model(first_ffa).sample
                    val_recon = torch.cat([first_ffa, val_recon], dim=-1)
                    val_recon = make_grid(val_recon, nrow=1).unsqueeze(0)
                    log_image = {"Early": (val_recon+1)/2}
                    val_recon = val_model(second_ffa).sample
                    val_recon = torch.cat([second_ffa, val_recon], dim=-1)
                    val_recon = make_grid(val_recon, nrow=1).unsqueeze(0)
                    log_image["Late"] = (val_recon+1)/2
                    # save_image(log_image["Early"], j(save_path, f'epoch {epoch+1} Early'))
                    # save_image(log_image["Late"], j(save_path, f'epoch {epoch+1} Late'))
                    accelerator.trackers[0].log_images(log_image, epoch+1)
                
            if (epoch + 1) % save_interval == 0 or epoch == cf['num_epochs'] - 1:
                val_model.save_pretrained(j(Config.project_dir, 'model_save'))
                dis_save = j(Config.project_dir, 'dis_save')
                os.makedirs(dis_save, exist_ok=True)
                torch.save(discriminator.state_dict(), j(dis_save, 'dis.pth'))
                
            

if __name__ == '__main__':
    main()