To use single image first get image with transparent background from remove.bg or autoremov.com
Then run command - 

python run_single_image.py --img_path ./example1.png --prompt "product in a kitchen used in meal generation" --out_path ./example1_out.png

Now to create a zoom out video on this example1_out.png image run command - 

python run_video.py --img_path ./example1_out.png --prompt "product in a kitchen used in meal generation" --out_path ./example1_video.mp4
