#!/usr/bin/env python2
import os
import subprocess as sp

jj_path = '../../src/jj.py'
input_dir = '../input/imgs_gray/'

def batch_run(b, q, u):
    output_dir = 'gray_{b}_{q}_{u}'.format(b=b, q=q, u=u)
    sp.check_call(['mkdir', '-p', output_dir])
    for image in sorted(os.listdir('../input/imgs_gray/')):

        print 'Compressing %s' % (image,)
        sp.check_call([
            jj_path,
            '-b', str(b),
            '-q', str(q),
            '-u', str(u),
            '-c',
            os.path.join(input_dir, image),
            os.path.join(output_dir, image.replace('.png', '.j')),
        ])

        print 'Decompressing %s' % (image,)
        sp.check_call([
            jj_path,
            '-d',
            os.path.join(output_dir, image.replace('.png', '.j')),
            os.path.join(output_dir, image)
        ])

for i in range(10):
    b = 2 ** i
    print 'Block size = %u' % (b,)
    batch_run(b, 50, 2000)

for i in range(10):
    q = int(2 ** i * 12.5)
    print 'Quantization factor = %u' % (q,)
    batch_run(8, q, 2000)

for i in range(10):
    u = int(2 ** i * 31.25)
    print 'Quantization threshold = %u' % (u,)
    batch_run(8, 50, u)

