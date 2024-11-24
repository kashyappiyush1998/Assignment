from PIL import Image, ImageOps
import cv2
import os
import torch
import ffmpeg
import sys
import argparse

from diffusers import StableDiffusionInpaintPipeline

def main(img_path, prompt, out_path):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    prompt = prompt

    border_each_frame = (50, 50, 50, 50)

    image = Image.open(img_path).convert('RGB')
    mask_frame = Image.new('RGB', image.size)
    mask_frame = ImageOps.expand(mask_frame,border=border_each_frame,fill='white')
    image_frame = image.copy()

    os.makedirs("./frames/", exist_ok=True)

    for i in range(90):
        image_frame = image_frame.convert('RGB')
        image_frame = ImageOps.expand(image_frame,border=border_each_frame,fill='black')
        out_image = pipe(prompt=prompt, image=image_frame, mask_image=mask_frame).images[0]
        img_name = f"{i:04d}.png"
        frame_path = os.path.join("./frames/", img_name)
        out_image = out_image.resize(image.size)
        out_image.save(frame_path)
        image_frame = out_image.copy()

    (
        ffmpeg
        .input('/content/frames/*.png', pattern_type='glob', framerate=25)
        .output(out_path)
        .run()
    )

    os.rmdir("./frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)

    args = parser.parse_args()
    main(img_path=args.img_path, prompt=args.prompt, out_path=args.out_path)
    
