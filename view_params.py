import os 
import sys
from PIL import Image
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":
      fig,axs = plt.subplots(1,3)
      axs[0].set_axis_off()
      axs[0].set_title("Noise")
      axs[1].set_axis_off()
      axs[1].set_title("Diffusion")
      axs[2].set_axis_off()
      axs[2].set_title("Procedual")
      folders = os.listdir(f'./tests/')
      for f in folders:
            time.sleep(0.5)
            attributes = f.split('-')
            scale = round(float(attributes[1]))
            persistence = round(float(attributes[3]))
            octaves = round(float(attributes[5]))
            lacunarity = round(float(attributes[7]))
            fig.suptitle(f"Scale:{scale} Persistence:{persistence} Octaves:{octaves} Lacunarity:{lacunarity}")
            axs[0].imshow(Image.open(f"./tests/{f}/noise.png"))
            axs[1].imshow(Image.open(f"./tests/{f}/diffusion.png"))
            axs[2].imshow(Image.open(f"./tests/{f}/procedual.png"))
            plt.draw()  # Redraw the updated figure
            plt.pause(0.1)  # Brief pause to allow for the update