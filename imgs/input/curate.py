#!/usr/bin/env python2

"Curate images located in the orig_imgs/ directory."
from __future__ import print_function
import PIL.Image
import os

os.system('mkdir -p imgs_color/')
os.system('mkdir -p imgs_gray/')

for x in os.listdir('orig_imgs/'):
    print(x)
    img = PIL.Image.open('orig_imgs/' + x)
    n = min(img.width, img.height)
    if img.width > n:
        dw = (img.width - n) // 2
    else:
        dw = 0
    if img.height > n:
        dh = (img.height - n) // 2
    else:
        dh = 0
    img = img.crop([dw, dh, dw + n, dh + n])
    img = img.resize((1000, 1000), PIL.Image.ANTIALIAS)
    y = x.split('.')[0] + '.png'
    img.convert('RGB').save('imgs_color/' + y)
    img.convert('L').save('imgs_gray/' + y)

