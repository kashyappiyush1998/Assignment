from PIL import Image, ImageOps
import cv2
import os
import torch
import sys
import argparse

from diffusers import StableDiffusionInpaintPipeline

def main(img_path, prompt):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")

    border = (1000, 400, 1000, 200)

    image = Image.open(img_path)
    mask_image = image.split()[-1]
    mask_image = ImageOps.invert(mask_image)
    mask_image = ImageOps.expand(mask_image,border=border,fill='white')
    image = image.convert('RGB')
    image = ImageOps.expand(image,border=border,fill='black')

    prompt = prompt

    out_image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]

    out_image = out_image.resize(image.size)

    return out_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)

    args = parser.parse_args()
    out_image = main(img_path=args.img_path, prompt=args.prompt)

    out_image.save(args.out_path)

