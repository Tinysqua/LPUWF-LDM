from dataclasses import dataclass


@dataclass
class Config():
    train_bc = 1
    eval_bc = 2
    num_epochs = 2000
    data_path = ['data/train']
    eval_path = 'data/test'
    single_channel = True
    mode = 'double'
    val_length = 2000

    # train process configuration
    val_inter = 80
    save_inter = 100
    sample_size = (512, 512)

    # model parameters
    in_channels = 1
    out_channels = 1
    up_and_down = (128, 256, 512)
    num_res_layers = 2
    vae_path = '' # set it if you want to continue training
    dis_path = '' # Example: weights/exp_X_XX/gen_save/vae.pth

    autoencoder_warm_up_n_epochs = 100
    # accelerate config
    split_batches = False
    mixed_precision = 'fp16'
    log_with = 'tensorboard'
    project_dir = 'weights/exp_test'
    gradient_accumulation_steps = 1
