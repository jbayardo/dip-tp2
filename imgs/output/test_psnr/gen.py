from __future__ import print_function
import random
import PIL.Image
import subprocess as sp

jj_path = '../../../src/jj.py'

for k in range(1, 128):
    print(k)
    img = PIL.Image.new('L', (256, 256))
    for i in range(256):
        for j in range(256):
            img.putpixel(
               (i, j),
               random.randrange(128 - k, 128 + k)
            )
    img.save('img%.3u.png' % (k,))

    sp.check_call([
        'python',
        jj_path, '-c',
        'img%.3u.png' % (k,),
        'out_img%.3u.j' % (k,),
    ])

    sp.check_call([
        'python',
        jj_path, '-d',
        'out_img%.3u.j' % (k,),
        'out_img%.3u.png' % (k,),
    ])

