#!/usr/bin/env python2
#from __future__ import print_function
import os
import subprocess as sp

jj_path = '../../src/jj.py'
input_dir = '../input/imgs_gray/'


def batch_run(block_size, quant_coef, quant_threshold):
    output_dir = 'gray_{b}_{q}_{u}'.format(b=block_size, q=quant_coef, u=quant_threshold)

    try:
        os.mkdir(output_dir)
    except:
        pass

    for image in sorted(os.listdir('../input/imgs_gray/')):
        print('Compressing %s' % (image,))
        sp.check_call([
            jj_path,
            '-b', str(block_size),
            '-q', str(quant_coef),
            '-u', str(quant_threshold),
            '-c',
            os.path.join(input_dir, image),
            os.path.join(output_dir, image.replace('.png', '.j')),
        ])

        print('Decompressing %s' % (image,))
        sp.check_call([
            jj_path,
            '-d',
            os.path.join(output_dir, image.replace('.png', '.j')),
            os.path.join(output_dir, image)
        ])

#for i in range(2, 10):
#    b = 2 ** i
#    print 'Block size = %u' % (b,)
#    batch_run(b, 50, 2000)
#
#for i in range(2, 10):
#    q = int(2 ** i * 12.5)
#    print 'Quantization factor = %u' % (q,)
#    batch_run(8, q, 2000)
#
#for i in range(10):
#    u = int(2 ** i * 31.25)
#    print 'Quantization threshold = %u' % (u,)
#    batch_run(8, 50, u)
#
#for i in range(1, 11):
#    u = int(5 * i)
#    print 'Quantization threshold = %u' % (u,)
#    batch_run(8, 50, u)
#
#for q in range(13, 26):
#    print 'Quantization factor = %u' % (q,)
#    batch_run(8, q, 1000000)

#for i in range(3, 11):
#    u = 5 * i
#    print 'Quantization threshold = %u' % (u,)
#    batch_run(8, 25, u)

for u in [64, 128, 256, 512, 1024, 2048]:
    print 'Quantization threshold = %u' % (u,)
    batch_run(8, 25, u)

