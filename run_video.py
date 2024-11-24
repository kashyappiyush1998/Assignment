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

    border = 100
    border_each_frame = (border, border, border, border)

    image = Image.open(img_path).convert('RGB')
    mask_frame = Image.new('RGB', image.size)
    mask_frame = ImageOps.expand(mask_frame,border=border_each_frame,fill='white')
    image_frame = image.copy()

    os.makedirs("./frames/", exist_ok=True)

    for i in range(10):
        image_frame = image_frame.convert('RGB')
        image_frame = ImageOps.expand(image_frame,border=border_each_frame,fill='black')
        out_image = pipe(prompt=prompt, image=image_frame, mask_image=mask_frame).images[0]
        img_name = f"{i:05d}.png"
        frame_path = os.path.join("./frames/", img_name)
        out_image = out_image.resize(image.size)
        out_image.save(frame_path)
        image_frame = out_image.copy()

    os.makedirs("./interpolated_frames/", exist_ok=True)
    
    imgs_path = [os.path.join("./frames/", i) for i in sorted(os.listdir("./frames/"))]
    total_count = 0
    for count in range(len(imgs_path)-1):
        img1 = Image.open(imgs_path[count])
        img2 = Image.open(imgs_path[count+1])
        
        img2 = img2.resize((img1.size[0]+border*2, img1.size[1]+border*2))
        
        for i in range(int(border/2)):
            rect = ((border-i*2), (border-i*2), img2.size[0]-(border-i*2), img2.size[1]-(border-i*2))
            img = img2.crop(rect)
            img = img.resize(img1.size)
            img_name = f"{total_count:05d}.png"
            img_path = os.path.join("./interpolated_frames/", img_name)
            img.save(img_path)
            total_count += 1

    (
        ffmpeg
        .input('./interpolated_frames/*.png', pattern_type='glob', framerate=25)
        .output(out_path)
        .run()
    )

    # os.rmdir("./frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)

    args = parser.parse_args()
    main(img_path=args.img_path, prompt=args.prompt, out_path=args.out_path)
    
