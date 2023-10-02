from tqdm.notebook import tqdm
import torch
from mydiffusers import StableDiffusionPipeline,StableDiffusionDepth2ImgPipeline,DDIMScheduler
import numpy as np
import abc
import shutil
from PIL import Image
from pytorch_lightning import seed_everything
import os
import cv2
import contextlib
import imageio
import argparse
import configparser

from video_model_cross_frame_attn import *
from file_utils import *
from common import *

def create_video_with_images(glob_pattern: str, output_filename: str):
    """
    Create a video from a set of images using ffmpeg.
    Images are gathered using a glob pattern (e.g. "img_????.png").
    """
    os.system(f"ffmpeg -pattern_type glob -i '{glob_pattern}' -c:v libx264 -r 30 -pix_fmt yuv420p {output_filename}")

def create_gif_with_images(glob_pattern: str, output_filename: str):
    """
    Create a video from a set of images using ffmpeg.
    Images are gathered using a glob pattern (e.g. "img_????.png").
    """
    os.system(f"ffmpeg -f image2 -framerate 30 -pattern_type glob -i '{glob_pattern}' -loop -1 {output_filename}")

########Stable Diffusion parameters##########
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = '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'
LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
model_id = "stabilityai/stable-diffusion-2-depth"
ldm_stable = StableDiffusionDepth2ImgPipeline.from_pretrained(model_id, use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
tokenizer = ldm_stable.tokenizer

seed = 2147483647
seed_everything(seed)

strength = 1.0

parser = argparse.ArgumentParser()

#######This file is to test the setup where we do not do any finetuning and we use the cfg update############
parser.add_argument("--config", type=str, default='/home/ceylan/pix2video/example_data/blackswan/blackswan_config_edit1.cfg',help="config file")
parser.add_argument("--input_path", type=str, default='/home/ceylan/pix2video/example_data/blackswan',help="input path")
parser.add_argument("--output_path", type=str, default='/home/ceylan/pix2video/example_data/blackswan_out',help="output path")
parser.add_argument("--inversion_path", type=str, default='/home/ceylan/pix2video/example_data/blackswan/inversion', help="inversion path")
parser.add_argument("--a_reverse", type=int, default=0,help="whether to process in reverse order")
parser.add_argument("--use_keyframe_only", type=int, default=0,help="whether to attend only to an anchor frame or not")
parser.add_argument("--classifier_guidance_flag", type=int, default=1,help="whether to do latent update")
parser.add_argument("--per_frame_classifier_guidance", type=int, default=1,help="whether to do cfg based update per frame")
parser.add_argument("--target_index", type=int, default=25,help="which target to match during latent update")
parser.add_argument("--use_self_attention_injection", type=int, default=1,help="whether to do feature injection or not")

opt = parser.parse_args()

config = configparser.ConfigParser()
config.read(opt.config)

folder_path = opt.input_path
output_path = opt.output_path
print("OUTPUT_PATH: ", output_path)
inversion_path = os.path.join(os.path.dirname(opt.config), 'inversion_no_finetune')
print("INVERSION_PATH: ", inversion_path)

noFrames = int(config['Arguments']['noFrames'])
noFrames=10
edit_prompt = config['Arguments']['edit_prompt']
org_prompt = config['Arguments']['org_prompt']
a_reverse = opt.a_reverse
use_keyframe_only = opt.use_keyframe_only
classifier_guidance_flag = opt.classifier_guidance_flag
per_frame_classifier_guidance = opt.per_frame_classifier_guidance
use_self_attention_injection = opt.use_self_attention_injection
target_index = opt.target_index

prompts = []
prompts.append(org_prompt)
prompts.append(edit_prompt)
edit_prompt_index = 0

if not os.path.exists(output_path):
    os.makedirs(output_path)

image_paths, depth_paths, noise_paths, output_paths, frame_numbers = get_data_paths(folder_path, output_path, noFrames, reverse=False)

images = []
depths = []

for i in range(noFrames):
    #load image
    input_image = load_512(image_paths[i])
    images.append(input_image)

    H = input_image.shape[0]
    W = input_image.shape[1]
    
    #load depth
    depth = load_depth(ldm_stable, depth_paths[i], input_image, W=W, H=H)
    depths.append(depth)

#create dataset
dataset = VideoDataset(ldm_stable, images, depths, folder_path, inversion_path, device, a_reverse)

#create model
video_model = VideoModel(ldm_stable, NUM_DDIM_STEPS, GUIDANCE_SCALE, output_path, [org_prompt], org_prompt, no_frames = len(images),
                        use_keyframe_only = use_keyframe_only, device=device,
                        classifier_guidance_flag = classifier_guidance_flag, per_frame_classifier_guidance=per_frame_classifier_guidance, 
                        use_self_attention_injection = use_self_attention_injection, target_index=target_index).to(device)

#perform inversion
video_model.perform_inversion(dataset)

video_model.test(dataset, output_path, [edit_prompt])

glob_pattern = os.path.join(output_path, './*.png')
output = os.path.join(output_path, 'out_cfg.gif')
create_gif_with_images(glob_pattern, output)
output = os.path.join(output_path, 'out_cfg.mp4')
create_video_with_images(glob_pattern, output)

for j in range(50):
    if j % 10 == 0:
        output = os.path.join(output_path, 'step_%02d/out.mp4' % j)
        glob_pattern = os.path.join(output_path, 'step_%02d/*.png' % j)
        create_video_with_images(glob_pattern, output)