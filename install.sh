#!/bin/bash

virtualenv --python=python3.6 env --prompt='(stylegan2-env) '

. env/bin/activate

pip install -r requirements.txt

mkdir -p models

# ffhq: high quality human faces trained from Flickr, https://github.com/NVlabs/ffhq-dataset
# Others are horse, cat, church, and car datasets from StyleGAN2.
for MODEL in stylegan2-ffhq-config-f.pkl stylegan2-horse-config-f.pkl stylegan2-cat-config-f.pkl stylegan2-church-config-f.pkl stylegan2-car-config-f.pkl; do
	curl -L "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/${MODEL}" -o models/${MODEL}
done

# Anime face model available here (no direct download): https://www.gwern.net/Faces#stylegan-2
