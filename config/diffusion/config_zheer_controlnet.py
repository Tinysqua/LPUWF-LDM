from dataclasses import dataclass


@dataclass
class Config():
    train_bc = 1
    eval_bc = 2
    num_epochs = 1000
    data_path = ['data/train']
    eval_path = 'data/test'
    single_channel = False
    mode = 'double'
    val_length = 2000

    # train process configuration
    val_inter = 100
    save_inter = 100
    sample_size = 512

    # aekl parameters
    in_channels = 1
    out_channels = 1
    up_and_down = (128, 256, 512)
    num_res_layers = 2
    scaling_factor = 0.18215
    vae_resume_path = '' # Example: weights/exp_X_XX/gen_save/vae.pth


    # stable_model parameters
    sd_num_channels = (128, 256, 512, 1024)
    attention_levels = (False, False, True, True)
    sd_resume_path = '' # Example: weights/exp_X_XX/model_save/model.pth
    controlnet_path = '' # Example: weights/exp_X_XX/model_save/controlnet.pth
    
    # controlnet_model parameters
    conditioning_embedding_num_channels = (32, 96, 256)
    diff_loss_coefficient = 0.25
    offset_noise = True
    
    # accelerate config
    split_batches = False
    mixed_precision = 'fp16'
    log_with = 'tensorboard'
    project_dir = 'weights/exp_X_XX'
