import torch
#################################### For Image ####################################
from PIL import Image
import numpy as np
import glob
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import os
import tqdm
# Load the model
def infer_img(image_path):
    image = Image.open(image_path)
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt="plant")
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]


    return masks, boxes, scores
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
in_dir = "/test/colmap/stage2/images"
out_dir = "/test/colmap/stage2/images"

jpgs = glob.glob(os.path.join(in_dir, "*.jpg"))
for jpg in tqdm.tqdm(jpgs):
    masks, boxes, scores = infer_img(jpg)
    mask = masks.sum(dim=0, keepdim=True)
    mask = mask.detach().cpu().numpy()
    mask = mask.astype(np.uint8)*255
    mask = Image.fromarray(mask[0, 0, :, :])
    filename = os.path.basename(jpg).split(".")[0]
    mask.save(os.path.join(out_dir, filename + "_mask.png"))


