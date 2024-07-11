import getpass, os
import io
import os
import warnings
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from PIL import Image , ImageFilter , ImageDraw, ImageFont
from ultralytics.utils import TQDM
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from stability_sdk import client
from torchvision.transforms import GaussianBlur
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation 
import os
import shutil
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import os
import gc
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import argparse

class BackgroundChangeGenerator:
    def __init(self):
        pass
    
    @staticmethod
    def overlay_mask(original_image, mask_image):
        original_image = original_image.convert('L').convert('RGB')
        mask_image = mask_image.convert('L')
        if original_image.size != mask_image.size or original_image.mode != mask_image.mode:
            mask_image = mask_image.resize(original_image.size)  # Resize the mask to match the original image
            mask_image = mask_image.convert(original_image.mode)  # Ensure the modes match
        opacity = 0.5
        overlay = Image.blend(original_image, mask_image, opacity)
        return overlay

    @staticmethod
    def get_masks(ann, image, better_quality=True):
        masks = ann.masks.data
        if better_quality:
            if isinstance(masks[0], torch.Tensor):
                masks = np.array(masks.cpu())
            for i, mask in enumerate(masks):
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                masks[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
        masks = np.around(masks)
        unified_mask = np.sum(masks, axis=0)
        unified_mask = (unified_mask / np.max(unified_mask)) * 255
        mask_image = unified_mask.astype(np.uint8)
        mask_image = Image.fromarray(mask_image, mode='L')
        overlay = BackgroundChangeGenerator.overlay_mask(image, mask_image)
        return overlay, mask_image

    @staticmethod
    def get_image(annotations):
        image = None
        pbar = TQDM(annotations, total=len(annotations))
        for ann in pbar:
            image = ann.orig_img[..., ::-1]  # BGR to RGB
            image = Image.fromarray(image)
            overlay, mask_image = BackgroundChangeGenerator.get_masks(ann, image)
        return image, overlay, mask_image

    @staticmethod
    def generate_background_change_api(prompt, image, mask, output_path, key, engine="stable-diffusion-xl-1024-v1-0"):
        os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
        os.environ['STABILITY_KEY'] = key
        stable_diffusion_output = None
        stability_api = client.StabilityInference(
            key=os.environ['STABILITY_KEY'],
            verbose=True,
            engine=engine,
        )

        answers = stability_api.generate(
            prompt=prompt,
            init_image=image,
            mask_image=mask,
            start_schedule=1,
            seed=44332211,
            steps=50,
            cfg_scale=8.0,
            width=1024,
            height=1024,
            sampler=generation.SAMPLER_K_DPMPP_2M
        )

        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        "Your request activated the API's safety filters and could not be processed."
                        "Please modify the prompt and try again.")
                if artifact.type == generation.ARTIFACT_IMAGE:
                    stable_diffusion_output = Image.open(io.BytesIO(artifact.binary))

        return stable_diffusion_output

    @staticmethod
    def generate_background_change_custom(prompt, image, mask, output_path,key,engine= "stabilityai/stable-diffusion-2-inpainting"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        access_token = key
        # Initialize the inpainting pipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            engine,
            torch_dtype=torch.float16,use_auth_token=access_token
        )
        pipe.to("cuda")
        torch.cuda.empty_cache()
        gc.collect()
        # Load the input image and mask image
        # Inpaint the image
        inpainted_image = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
        # Save the inpainted image to the specified output path
        return inpainted_image

    @staticmethod
    def get_downloads_path():
        if os.name == 'posix':  # Unix-based system (like Ubuntu)
            downloads_path = os.path.expanduser("~") + "/Downloads"
        elif os.name == 'nt':  # Windows
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        else:
            raise Exception("Unsupported operating system")
        return downloads_path
    
    @staticmethod
    def load_file_path():
        file_path = None
        root = tk.Tk()
        root.withdraw() 

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.gif *.bmp")])
        if file_path:
            print("Selected file path: " + file_path)
            root.destroy()
        else:
            print("No file selected")
            root.destroy()
        root.mainloop() 
        return file_path 
 
    @staticmethod
    def draw_rectangle(event, x, y, flags, param):
        global x1, y1, x2, y2, drawing, image
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x1, y1 = x, y
            x2, y2 = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x2, y2 = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

        img_copy = image.copy()
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Image', img_copy)

    @staticmethod
    def canvas(image_path):
        global x1, y1, x2, y2, drawing, image
        drawing = False
        file_path = None
        x1, y1, x2, y2 = -1, -1, -1, -1
        image = None
        if image_path is None:
            file_path = BackgroundChangeGenerator.load_file_path()
            image = cv2.imread(file_path)
            cv2.namedWindow('Image')
            cv2.setMouseCallback('Image', BackgroundChangeGenerator.draw_rectangle)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Press 'ESC' to exit
                    break
            cv2.destroyAllWindows()
        else:
            file_path = image_path
            print("Selected file path: " + file_path)
            image = cv2.imread(file_path)
            cv2.namedWindow('Image')
            cv2.setMouseCallback('Image', BackgroundChangeGenerator.draw_rectangle)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Press 'ESC' to exit
                    break
            cv2.destroyAllWindows()      
        return file_path,[x1, y1, x2, y2]
    
    @staticmethod
    def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
        min_height = min(im.height for im in im_list)
        im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height),resample=resample)
                        for im in im_list]
        total_width = sum(im.width for im in im_list_resize)
        dst = Image.new('RGB', (total_width, min_height))
        pos_x = 0
        for im in im_list_resize:
            dst.paste(im, (pos_x, 0))
            pos_x += im.width
        return dst
    
    @staticmethod
    def mask_the_region(model,image_path):
        image_path , box = BackgroundChangeGenerator.canvas(image_path)
        everything_results = model(image_path, device='cuda', retina_masks=True, imgsz=640, conf=0.4, iou=0.9)
        prompt_process = FastSAMPrompt(image_path, everything_results, device='cuda')
        ann = prompt_process.box_prompt(bbox=box)
        input_image,annotated_image,mask = BackgroundChangeGenerator.get_image(ann)
        draw = ImageDraw.Draw(input_image)
        draw.rectangle(box, outline="green", width=2)
        draw_ann = ImageDraw.Draw(annotated_image)
        draw_ann.rectangle(box, outline="green", width=2)
        BackgroundChangeGenerator.get_concat_h_multi_resize([input_image, annotated_image, mask]).show()
        BackgroundChangeGenerator.get_concat_h_multi_resize([input_image, annotated_image, mask]).save('./concat.png')
        return input_image,annotated_image,mask
    
    @staticmethod
    def main():
        bg_change_generator = BackgroundChangeGenerator()
        model = FastSAM('FastSAM-s.pt') 
        print('Model Loaded !!!!!')
    
        input_image = None
        annotated_image = None
        mask = None
        
        parser = argparse.ArgumentParser(description="Select a file from the command line")
        parser.add_argument("--path", type=str, help="Path to the file you want to use")
        parser.add_argument('--custom', default=False, action='store_true', help='to utilize custom weights')

        args = parser.parse_args()
        if args.path:
            input_image, annotated_image, mask = bg_change_generator.mask_the_region(model, args.path)
        else:
            input_image, annotated_image, mask = bg_change_generator.mask_the_region(model, None)
        
        prompt = input("Enter the prompt to change the background: ")

        if args.custom:
            # If you have a GPU with a minimum of 8GB memory, use this function
            stable_diffusion_output = bg_change_generator.generate_background_change_custom(
                prompt=prompt,
                image=input_image,
                mask=mask,
                output_path="./out.png",
                key='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',
                engine="stabilityai/stable-diffusion-2-inpainting"
            )
        else:
            stable_diffusion_output = bg_change_generator.generate_background_change_api(
                prompt=prompt,
                image=input_image,
                mask=mask,
                output_path="./out.png",
                key='sk-y1SMimJTiCmiW7RvHnSYxZGb8zbo2jr4xdmW5M8eOs9y172D',
                engine="stable-diffusion-xl-1024-v1-0"
            )
        
        stable_diffusion_output.save('stable_diffusion_output.png')
        BackgroundChangeGenerator.get_concat_h_multi_resize([input_image, annotated_image, mask, stable_diffusion_output]).show()
        BackgroundChangeGenerator.get_concat_h_multi_resize([input_image, annotated_image, mask, stable_diffusion_output]).save('./results.png')
            
        
if __name__ == "__main__":
    bg_change_generator = BackgroundChangeGenerator()
    bg_change_generator.main()
   
