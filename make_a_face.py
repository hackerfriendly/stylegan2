#!/usr/bin/env python3

# Heavily adapted from run_generator.py

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import json

import pretrained_networks

#----------------------------------------------------------------------------

def generate_images(model, seed, psi, out):
    print('Loading networks from "%s"...' % model)
    _G, _D, Gs = pretrained_networks.load_networks(model)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if psi is not None:
        Gs_kwargs.psi = psi

    print(f'Generating image for seed {seed}')
    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]

    PIL.Image.fromarray(images[0], 'RGB').save(out, compression='JPEG', quality=85)
    with open(f"{out.rstrip('.jpg')}.json", "w", encoding="UTF-8") as f:
        f.write(json.dumps(
            {
                "engine": "StyleGAN2",
                "seed": seed,
                "psi": psi,
                "model": model
            }
        ))

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 face generator.'''
    )

    parser.add_argument('--model', help='Model filename', required=True)
    parser.add_argument('--seed', type=int, help='Random seed', required=True)
    parser.add_argument('--psi', type=float, help='Truncation psi (default: %(default)s)', default=0.6)
    parser.add_argument('--out', help='Output path for jpg and json (default: %(default)s)', default='out.jpg')

    args = parser.parse_args()

    generate_images(args.model, args.seed, args.psi, args.out)

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
