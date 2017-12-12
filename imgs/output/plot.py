#!/usr/bin/env python2
# coding:utf-8:
#from __future__ import unicode_literals

import os
import sys
import matplotlib.pyplot as plt
import subprocess as sp

jj_path = '../../src/jj.py'
input_dir = '../input/imgs_gray/'
plot_dir = 'gray_plots'

def compression_rates(b, q, u):
    output_dir = 'gray_{b}_{q}_{u}'.format(b=b, q=q, u=u)
    rates = []
    for image in os.listdir(input_dir):
        img1 = os.path.join(input_dir, image)
        img2 = os.path.join(output_dir, image.replace('.png', '.j'))
        if not os.path.exists(img2):
            continue
        rate = sp.check_output([jj_path, '--rate', img1, img2])
        rates.append(float(rate))
    return sorted(rates)

def peak_signal_to_noise_ratio(b, q, u):
    output_dir = 'gray_{b}_{q}_{u}'.format(b=b, q=q, u=u)
    psnrs = []
    for image in os.listdir(input_dir):
        img1 = os.path.join(input_dir, image)
        img2 = os.path.join(output_dir, image)
        if not os.path.exists(img2):
            continue
        psnr = sp.check_output([jj_path, '--psnr', img1, img2])
        psnrs.append(float(psnr))
    return sorted(psnrs)

def plot(name, xlabel, ylabel, fn, bs, qs, us):
    print 'Graficando %s' % (name,)
    plt.clf()
    data = []
    for b in bs:
        for q in qs:
            for u in us:
                values = fn(b, q, u)
                data.append(values)

    # Hace varios box-plots
    plt.boxplot(data)

    # Poner etiquetas correctas en el eje X
    for xs in [bs, qs, us]:
        if len(xs) > 1:
            plt.xticks(
                list(range(1, len(xs) + 1)),
                [str(x) for x in xs]
            )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(plot_dir, name + '.png'))

bs = [2 ** i for i in range(10)]
qs = [int(2 ** i * 12.5) for i in range(7)]
us = [int(2 ** i * 31.25) for i in range(10)]

#plot(
#    'b_rate',
#    u'Tamaño de bloque (B)',
#    u'Tasa de compresión',
#    compression_rates,
#    bs, [50], [2000],
#)
#plot(
#    'b_psnr',
#    u'Tamaño de bloque (B)',
#    u'PSNR',
#    peak_signal_to_noise_ratio,
#    bs, [50], [2000],
#)
plot(
    'q_rate',
    u'Factor de cuantización (Q)',
    u'Tasa de compresión',
    compression_rates,
    [8], qs, [2000],
)
plot(
    'q_psnr',
    u'Factor de cuantización (Q)',
    u'PSNR',
    peak_signal_to_noise_ratio,
    [8], qs, [2000],
)
#plot(
#    'u_rate',
#    u'Umbral de cuantización (U)',
#    u'Tasa de compresión',
#    compression_rates,
#    [8], [50], us,
#)
#plot(
#    'u_psnr',
#    u'Umbral de cuantización (U)',
#    u'PSNR',
#    peak_signal_to_noise_ratio,
#    [8], [50], us,
#)
#
