import noise
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

from transformers import AutoImageProcessor, UperNetForSemanticSegmentation


shape = (1024,1024)
scale = 100.0
octaves = 6
persistence = 0.5
lacunarity = 2.0

world = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        world[i][j] = noise.pnoise2(i/scale, 
                                    j/scale, 
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity, 
                                    repeatx=1024, 
                                    repeaty=1024, 
                                    base=0)
world_norm = (world  - world.min()) / (world.max() - world.min()) * 255
world_norm = world_norm.astype(np.uint8)
img = Image.fromarray(world_norm, mode='L')
img.save('./images/noise.png')


blue = [65,105,225]
green = [34,139,34]
beach = [238, 214, 175]

def add_color(world):
      b = 0
      g = 0
      bea = 0
      color_world = np.zeros(world.shape+(3,))
      for i in range(shape[0]):
            for j in range(shape[1]):
                  if world[i][j] < -0.05:
                        color_world[i][j] = blue
                        b += 1
                  elif world[i][j] < 0:
                        color_world[i][j] = beach
                        bea += 1
                  elif world[i][j] < 1.0:
                        color_world[i][j] = green
                        g += 1
      return color_world

color_world = add_color(world)
color_world = color_world.astype(np.uint8)
img_color = Image.fromarray(color_world, mode='RGB')
img.save('./images/procedual.png')



controlnet = ControlNetModel.from_pretrained(
"lllyasviel/sd-controlnet-seg"
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)



image_controlnet = pipe("Create a landscape of ocean, land, and beach using only the colors {blue}, {green}, and {beach} for blue green and yellow and no other colors. Emualte the style of a 8bit game", img, num_inference_steps=20).images[0]
image_controlnet.save('./images/diffusion.png')