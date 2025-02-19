import sys;sys.path.append('./')

from tqdm import tqdm

from utils.Condition_aug_dataloader import double_form_dataloader

import torch
from torch.nn import functional as F
from torch import nn
from generative.networks.nets import AutoencoderKL
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet, ControlNet
from generative.networks.schedulers import DDPMScheduler

from utils.common import get_parameters, save_config_to_yaml, one_to_three
from config.diffusion.config_zheer_controlnet import Config
from os.path import join as j
from accelerate import Accelerator
from torchvision.utils import make_grid, save_image
import os

def main():
    cf = save_config_to_yaml(Config, Config.project_dir)
    accelerator = Accelerator(**get_parameters(Accelerator, cf))
    train_dataloader = double_form_dataloader(Config.data_path, 
                                        Config.sample_size, 
                                        Config.train_bc, 
                                        Config.mode, 
                                        read_channel='gray')
    device = 'cuda'
    val_dataloader = double_form_dataloader(
        Config.eval_path, 
        Config.sample_size, 
        Config.eval_bc, 
        Config.mode, 
        read_channel='gray')
    
    device = 'cuda'
    attention_levels = (False, ) * len(Config.up_and_down)
    vae = AutoencoderKL(
        spatial_dims=2, 
        in_channels=Config.in_channels, 
        out_channels=Config.out_channels, 
        num_channels=Config.up_and_down, 
        latent_channels=4,
        num_res_blocks=Config.num_res_layers, 
        attention_levels = attention_levels
        )
    vae = vae.eval().to(device)
    if len(Config.vae_resume_path):
        vae.load_state_dict(torch.load(Config.vae_resume_path))

    model = DiffusionModelUNet(
    num_res_blocks=2, 
    spatial_dims=2,
    in_channels=4,
    out_channels=4,
    num_channels=Config.sd_num_channels,
    attention_levels=Config.attention_levels,
    )
    if len(Config.sd_resume_path):
        model.load_state_dict(torch.load(Config.sd_resume_path))
    model = model.to(device)

    controlnet = ControlNet(
        spatial_dims=2, 
        in_channels=4, 
        num_res_blocks=2, 
        num_channels=Config.sd_num_channels, # keep consistent with diffusion model
        attention_levels=Config.attention_levels, 
        conditioning_embedding_in_channels=3, 
        conditioning_embedding_num_channels=Config.conditioning_embedding_num_channels
    ).to(device)

    if len(Config.controlnet_path):
        controlnet.load_state_dict(torch.load(Config.controlnet_path), strict=False)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer_con = torch.optim.Adam(params=controlnet.parameters(), lr=2.5e-5)
    optimizer_sd = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
    inferer = DiffusionInferer(scheduler)
    weighted_mse_loss = nn.MSELoss(reduction='none')

    val_interval = Config.val_inter
    save_interval = Config.save_inter

    if len(Config.log_with):
        accelerator.init_trackers('train_example')

    global_step = 0
    latent_shape = None
    # scaling_factor = 1 / torch.std(next(iter(train_dataloader))[0][1])
    scaling_factor = Config.scaling_factor
    for epoch in range(Config.num_epochs):
        model.train()
        controlnet.train()
        epoch_loss = 0
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch+1}")
        for step, batch in enumerate(train_dataloader):
            batch = batch[0]
            slo, early_ffa, late_ffa = batch[0].to(device), batch[1].to(device), batch[2].to(device) # train for ffa
            ffa_diff = torch.abs(early_ffa - late_ffa)
            optimizer_con.zero_grad(set_to_none=True)
            optimizer_sd.zero_grad(set_to_none=True)
            with torch.no_grad():
                late_ffa = vae.encode_stage_2_inputs(late_ffa)
                late_ffa = late_ffa * scaling_factor
            ffa_diff = F.interpolate(ffa_diff, size=late_ffa.shape[-2:])
            latent_shape = list(late_ffa.shape);latent_shape[0] = Config.eval_bc
            if Config.offset_noise:
                noise = torch.randn_like(late_ffa) + 0.1 * torch.randn(late_ffa.shape[0], late_ffa.shape[1], 1, 1).to(late_ffa.device)
            else:
                noise = torch.randn_like(late_ffa)

            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (late_ffa.shape[0],), device=late_ffa.device
            ).long()
            ffa_noised = scheduler.add_noise(late_ffa, noise, timesteps)
            down_block_res_samples, mid_block_res_sample = controlnet(
                x=ffa_noised, timesteps=timesteps, controlnet_cond=slo
            )
            noise_pred = model(
                x=ffa_noised,
                timesteps=timesteps,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
            loss = F.mse_loss(noise_pred.float(), noise.float()) + Config.diff_loss_coefficient*(ffa_diff*weighted_mse_loss(noise_pred.float(), noise.float())).mean()
            loss.backward()
            optimizer_con.step()
            optimizer_sd.step()
            epoch_loss += loss.item()
            logs = {"loss": epoch_loss / (step + 1)}
            progress_bar.update()
            progress_bar.set_postfix(logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if (epoch + 1) % val_interval == 0 or epoch == cf['num_epochs'] - 1:
            model.eval()
            controlnet.eval()
            batch = next(iter(val_dataloader))
            batch = batch[0]
            slo, early_ffa, late_ffa = batch[0].to(device), batch[1].to(device), batch[2].to(device) 
            noise = torch.randn(latent_shape).to(device)
            progress_bar_sampling = tqdm(scheduler.timesteps, total=len(scheduler.timesteps), ncols=110, position=0, leave=True)
            with torch.no_grad():
                for t in progress_bar_sampling:
                    down_block_res_samples, mid_block_res_sample = controlnet(
                    x=noise, timesteps=torch.Tensor((t,)).to(device).long(), controlnet_cond=slo
                    )
                    noise_pred = model(
                    noise,
                    timesteps=torch.Tensor((t,)).to(device),
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    )
                    noise, _ = scheduler.step(model_output=noise_pred, timestep=t, sample=noise)

            
            with torch.no_grad():
                image = vae.decode_stage_2_outputs(noise / scaling_factor)
            image, late_ffa = one_to_three(image), one_to_three(late_ffa) 
            image = torch.cat([slo, image, late_ffa], dim=-1)
            image = (make_grid(image, nrow=1).unsqueeze(0)+1)/2
            log_image = {"Late": image.clamp(0, 1)}
            save_path = j(Config.project_dir, 'image_save')
            os.makedirs(save_path, exist_ok=True)
            save_image(log_image["Late"], j(save_path, f'epoch_{epoch + 1}_Late.png'))

            # accelerator.trackers[0].log_images(log_image, epoch+1)

        if (epoch + 1) % save_interval == 0 or epoch == cf['num_epochs'] - 1:
            save_path = j(Config.project_dir, 'model_save')
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), j(save_path, 'model.pth'))
            torch.save(controlnet.state_dict(), j(save_path, 'controlnet.pth'))

if __name__ == '__main__':
    main()