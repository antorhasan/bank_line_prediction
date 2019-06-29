from PIL import Image
import numpy as np
img = Image.open('./data/4.tif')
im = np.array(img)
NewValue = (((im + 32768) * (255- 0)) / (32768 + 32768)) + 0
print(im)
print(im.shape)
print(NewValue)
