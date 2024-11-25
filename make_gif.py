import imageio
import os

start = 0
end = 99


images = []
for i in range(start, end+1):
    images.append(imageio.imread(f"images/diffusion{i}.png"))
imageio.mimsave("images/animation.gif", images, duration=0.1)  # 0.1 seconds per frame

