from PIL import Image
import os
from tqdm import tqdm

start = 0
end = 199


images = []
for i in tqdm(range(start, end+1)):
    images.append(Image.open(f"images/diffusion{i}.png"))
    
images[0].save('pillow_imagedraw.gif', 
               save_all = True, append_images = images[1:], 
               optimize = True, duration = 0.1)
    

