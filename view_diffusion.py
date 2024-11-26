import os 
import sys
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np

if __name__ == "__main__":
      fig = plt.figure()
      
      images = np.sort(os.listdir(f'./images/'))
      max_num = 0
      for img in images:
            if('diffusion' in img):
                  num = (int(img[9:-4]))
                  max_num = max(num, max_num)
      for i in range(max_num):
            plt.clf()  # Clear the figure to remove previous content
            plt.title(f"Diffusion Step {i}")
            plt.axis('off')
            plt.imshow(Image.open(f"./images/diffusion{i}.png"))
            plt.draw()  # Redraw the updated figure
            plt.pause(0.1)  # Brief pause to allow for the update
            time.sleep(0.1)
            