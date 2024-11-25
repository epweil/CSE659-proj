import noise
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from tqdm import tqdm


start = 49
runs = 50


controlnet = ControlNetModel.from_pretrained(
"lllyasviel/sd-controlnet-seg",
cache_dir="./models"
).to("cuda")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None,
cache_dir="./models"
).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

img_in = Image.open(f'./images/diffusion{start}.png')
for i in tqdm(range(start+1, start+runs+1)):
      img_in = pipe("Create a landscape of ocean, land, and beach using only the colors {blue}, {green}, and {beach} for blue green and yellow and no other colors. Emualte the style of a 8bit game", img_in, num_inference_steps=20, VERBOSE = False).images[0]
      img_in.save(f'./images/diffusion{i}.png')