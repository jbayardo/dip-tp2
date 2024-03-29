#!/usr/bin/env python2
# coding:utf-8:
# from __future__ import unicode_literals
from __future__ import print_function
import subprocess as sp

import matplotlib.pyplot as plt

jj_path = '../../../src/jj.py'


def peak_signal_to_noise_ratio(k):
    psnr = sp.check_output([
        'python',
        jj_path, '--psnr',
        'img%.3u.png' % (k,),
        'out_img%.3u.png' % (k,),
    ])
    return float(psnr)


plt.clf()

for k in range(1, 128):
    print(k)
    plt.plot(k, peak_signal_to_noise_ratio(k), 'o', color='blue')

plt.xlabel('k')
plt.ylabel('PSNR')
plt.savefig('k_vs_psnr.png')
plt.show()
