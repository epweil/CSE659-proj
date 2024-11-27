import noise
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import sys
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from tqdm import tqdm
import os

blue = (65,105,225)
green = (34,139,34)
beach = (238, 214, 175)
snow = (255, 250, 250)
mountain = (139, 137, 137)

def add_color(world):
      color_world = np.zeros(world.shape+(3,))
      for i in range(shape[0]):
            for j in range(shape[1]):
                  if world[i][j] < -0.05:
                        color_world[i][j] = blue
                  elif world[i][j] < 0:
                        color_world[i][j] = beach
                  elif world[i][j] < 0.1:
                        color_world[i][j] = green
                  elif world[i][j] < 0.35:
                        color_world[i][j] = mountain
                  elif world[i][j] < 1.0:
                        color_world[i][j] = snow
      return color_world
                



if __name__ == "__main__":
      controlnet = ControlNetModel.from_pretrained(
                                    "lllyasviel/sd-controlnet-hed",
                                    cache_dir="./models"
                                    ).to("cuda")
      pipe = StableDiffusionControlNetPipeline.from_pretrained(
      "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None,
      cache_dir="./models"
      ).to("cuda")
      pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
      pipe.enable_progress_bar = False
      for scale in tqdm(np.linspace(100,512, 4)):
            for persistence in np.linspace(1,10, 4):
                  for octaves in np.linspace(2,10, 4):
                        for lacunarity in np.linspace(1,10, 4):
                              proj_name = f"scale-{scale}-persistence-{persistence}-octaves-{octaves}-lacunarity-{lacunarity}"  
                              dir = f'./tests_hed/{proj_name}/'  
                              if( not os.path.exists(dir)):
                                    os.mkdir(dir)
                                    shape = (512,512)
                                    world = np.zeros(shape)
                                    for i in range(shape[0]):
                                          for j in range(shape[1]):
                                                world[i][j] = noise.pnoise2(i/scale, 
                                                                              j/scale, 
                                                                              octaves=int(octaves), 
                                                                              persistence=persistence, 
                                                                              lacunarity=lacunarity, 
                                                                              repeatx=shape[0], 
                                                                              repeaty=shape[1], 
                                                                              base=0)
                                    world_norm = (world  - world.min()) / (world.max() - world.min()) * 255
                                    world_norm = world_norm.astype(np.uint8)
                                    img = Image.fromarray(world_norm, mode='L')
                                    img.save(f'{dir}/noise.png')


                                    blue = [65,105,225]
                                    green = [34,139,34]
                                    beach = [238, 214, 175]

                                    color_world = add_color(world)
                                    color_world = color_world.astype(np.uint8)
                                    img_color = Image.fromarray(color_world, mode='RGB')
                                    img_color.save(f'{dir}/procedual.png')

                                    img_diffusion = pipe("Create a two dimensional game region of the ocean, beach, mountains, grass, and snow.", img, num_inference_steps=20).images[0]
                                    img_diffusion.save(f'{dir}/diffusion.png')
