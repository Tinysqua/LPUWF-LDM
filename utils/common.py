import yaml
import inspect
from shutil import copyfile, copy
import os
import torch
from torchvision import transforms
from diffusers import ControlNetModel


def copy_yaml_to_folder(yaml_file, folder):
    """
    将一个 YAML 文件复制到一个文件夹中
    :param yaml_file: YAML 文件的路径
    :param folder: 目标文件夹路径
    """
    # 确保目标文件夹存在
    os.makedirs(folder, exist_ok=True)

    # 获取 YAML 文件的文件名
    file_name = os.path.basename(yaml_file)

    # 将 YAML 文件复制到目标文件夹中
    copy(yaml_file, os.path.join(folder, file_name))

def force_remove_empty_dir(path):
    try:
        os.rmdir(path)
        print(f"Directory '{path}' removed successfully.")
    except FileNotFoundError:
        print(f"Directory '{path}' not found.")
    except OSError as e:
        print(f"Error removing directory '{path}': {e}")

def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        for key in config.keys():
            if type(config[key]) == list:
                config[key] = tuple(config[key])
        return config
    
def get_parameters(fn, original_dict):
    new_dict = dict()
    arg_names = inspect.getfullargspec(fn)[0]
    for k in original_dict.keys():
        if k in arg_names:
            new_dict[k] = original_dict[k]
    return new_dict

def write_config(config_path, save_path):
    copyfile(config_path, save_path)
    
@torch.no_grad()
def clip_encode_images(image, feature_extractor, image_encoder):
    '''
    Default that all the variable here is in the same device
    '''
    dtype = next(image_encoder.parameters()).dtype
    resize = transforms.Resize(224)
    # image = feature_extractor(images=image, return_tensors="pt").pixel_values
    image_embeddings = image_encoder(resize(image)).image_embeds
    image_embeddings = image_embeddings.unsqueeze(1)
    return image_embeddings
    
def check_dir(dire):
    if not os.path.exists(dire):
        os.makedirs(dire)
    return dire

def control_net_from_unet_by_hand(unet, 
                                  conditioning_channels=3,
                                  controlnet_conditioning_channel_order: str = "rgb",
                                  conditioning_embedding_out_channels = (16, 32, 96, 256)) -> ControlNetModel:
    
    controlnet = ControlNetModel(
            conditioning_channels = conditioning_channels,
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        )
    controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
    controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
    controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
    
    if unet.class_embedding:
        controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

    controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
    controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

    return controlnet