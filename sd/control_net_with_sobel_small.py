import sys; sys.path.append('./')
from utils.Condition_dataloader import double_form_dataloader
import argparse
from diffusers import (DDPMScheduler, UNet2DConditionModel, 
                       AutoencoderKL, ControlNetModel, 
                       StableDiffusionControlNetPipeline)
import torch
from utils.common import (load_config, get_parameters, 
                          copy_yaml_to_folder, control_net_from_unet_by_hand)
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from os.path import join as j
from transformers import CLIPTextModel, CLIPTokenizer


def main(args):
    cf = load_config(args.config_path)
    copy_yaml_to_folder(args.config_path, cf['project_dir'])
    train_dataloader = double_form_dataloader(cf['data_path'], 
                                        cf['img_sz'], 
                                        cf['train_bc'], 
                                        'first', 
                                        data_aug=False, 
                                        let_in_sobel=True)
    val_dataloader = double_form_dataloader(cf['eval_path'], 
                                        cf['img_sz'], 
                                        cf['eval_bc'], 
                                        'first', 
                                        data_aug=False, 
                                        let_in_sobel=True)
    accelerator = Accelerator(**get_parameters(Accelerator, cf))
    autoencoderkl_tar = AutoencoderKL.from_pretrained(cf['target_path'])
    autoencoderkl_tar = autoencoderkl_tar.eval().to(accelerator.device)
    
    
    tokenizer = CLIPTokenizer.from_pretrained(cf['stable_path'], subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(cf['stable_path'], subfolder="text_encoder", use_safetensors=False)
    text_encoder = text_encoder.to(accelerator.device)
    prompt_word = 'To generate an Ultra-Wide-angle Fundus Fluorescein Angiography from an Ultra-Wide-angle Fundus Scanning Laser Ophthalmoscopy'
    prompt = [prompt_word] * cf['train_bc']
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]
    print("text_shape: ", text_embeddings.shape)
    
    model = UNet2DConditionModel(sample_size=cf["sample_size"], 
                        in_channels=cf['in_channels'],
                        out_channels=cf['out_channels'], 
                        layers_per_block=cf['layers_per_block'],
                        block_out_channels=cf['block_out_channels'],
                        down_block_types=cf['down_block_types'], 
                        up_block_types=cf['up_block_types'], 
                        cross_attention_dim=cf['cross_attention_dim'])
    if len(cf['unet_resume']):
        model = UNet2DConditionModel.from_pretrained(cf['unet_resume'])
        
    controlnet = control_net_from_unet_by_hand(model, conditioning_channels=4)
    
    model.requires_grad_(False)
    autoencoderkl_tar.requires_grad_(False)
    text_encoder.requires_grad_(False)
    model.to(accelerator.device)
    
    noise_scheduler = DDPMScheduler(**get_parameters(DDPMScheduler, cf))
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=cf["lr"])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * cf['num_epochs']),
    )
    
    val_interval = cf['val_inter']
    save_interval = cf['save_inter']
    if len(cf['log_with']):
        accelerator.init_trackers('train_example')
    controlnet, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler, val_dataloader
        )
    global_step = 0
    scaling_factor = autoencoderkl_tar.config.scaling_factor
    for epoch in range(cf['num_epochs']):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            batch = batch[0]
            slo, early_ffa = batch[0], batch[1]
            targets = early_ffa
            conditions = slo
            with accelerator.accumulate(controlnet):
                latents = autoencoderkl_tar.encode(targets).latent_dist.sample()
                # print('latent shape: ', latents.shape)
                latents = latents * scaling_factor

                bs = latents.shape[0]
                noise = torch.randn_like(latents, device=latents.device)
                
                timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, 
                                                noise, timesteps)
                # print('Shape: ', noisy_latents.shape)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=conditions,
                    return_dict=False, 
                )
                model_pred = model(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduce='mean')
                
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(controlnet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs, refresh=False)
            progress_bar.update(1)
            accelerator.log(logs, step=global_step)
            global_step += 1
            
        if accelerator.is_main_process:
            
            pipeline = StableDiffusionControlNetPipeline(autoencoderkl_tar, 
                                                         text_encoder, 
                                                         tokenizer, 
                                                         model,
                                                         accelerator.unwrap_model(controlnet), 
                                                         noise_scheduler, 
                                                         None, 
                                                         None,
                                                         False
                                                         )
            prompt = [prompt_word] * cf['eval_bc']
            if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
                val_data = next(iter(val_dataloader))[0]
                targets, conditions = val_data[1], val_data[0]
                images = pipeline(prompt, conditions, 
                                  num_inference_steps=100, guidance_scale=1.0, 
                                  output_type='pt').images 
                conditions = conditions[:,:3,...]
                images = torch.cat([conditions, (images-0.5)/0.5, targets], dim=-1)
                images = (make_grid(images, nrow=2).unsqueeze(0) + 1) / 2 
                accelerator.trackers[0].log_images({'Generate': images.cpu().detach()}, epoch+1) 
                
            if (epoch + 1) % save_interval == 0 or epoch == cf['num_epochs'] - 1:
                if cf['only_save_part']:
                    accelerator.unwrap_model(controlnet).save_pretrained(cf['project_dir']) 
                    noise_scheduler.save_pretrained(cf['project_dir']) 
                else:
                    pipeline.save_pretrained(cf['project_dir'])   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/control_net_with_sobel_config.yaml')

    args = parser.parse_args()
    main(args)