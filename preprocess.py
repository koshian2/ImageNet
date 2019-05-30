from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def imagenet_data_augmentation(pillow_img, target_size,
                               area_min=0.08, area_max=1.0,
                               aspect_min=0.75, aspect_max=4.0/3.0):
    """
    Data augmentation for single image(based on GoogLe Net)
    # input : pillow_img = PIL instance
    #       : target_size = resized width / height
    # output : uint8 numpy array
    # optional : cropped area = U([area_min, area_max])
                 cropped aspect ratio = U([aspect_min, aspect_max])
    """
    # aspect_ratio = width / height
    # cropped_width = sqrt(S*a)
    # cropped_height = sqrt(S/a)
    original_width, original_height = pillow_img.size
    cropped_area = np.random.uniform(area_min, area_max) * original_width * original_height
    cropped_aspect_ratio = np.random.uniform(aspect_min, aspect_max)
    cropped_width = int(np.sqrt(cropped_area * cropped_aspect_ratio))
    cropped_height = int(np.sqrt(cropped_area / cropped_aspect_ratio))
    # crop left / right point
    if original_width > cropped_width:
        horizontal_slide = int(np.random.uniform(0, original_width-cropped_width))
        left, right = horizontal_slide, horizontal_slide+cropped_width
    else:
        horizontal_slide = (cropped_width - original_width) // 2
        left, right = -horizontal_slide, horizontal_slide+original_width
    # crop top / bottom point
    if original_height > cropped_height:
        vertical_slide = int(np.random.uniform(0, original_height-cropped_height))
        top, bottom = vertical_slide, vertical_slide+cropped_height
    else:
        vertical_slide = (cropped_height - original_height) // 2
        top, bottom = -vertical_slide, vertical_slide+original_height
    cropped = pillow_img.crop((left, top, right, bottom))
    resized = cropped.resize((target_size, target_size), Image.LINEAR)
    # horizontal flip
    if np.random.random() >= 0.5:
        resized = ImageOps.mirror(resized)
    # auto contrast (a bit slow)
    if np.random.random() >= 0.5:
        resized = ImageOps.autocontrast(resized, 
                        np.random.uniform(0, 1.0), ignore=0) # ignore black background
    return np.asarray(resized, np.uint8)

def validation_image_load(pillow_img, target_size):
    """
    Convert pillow instance to numpy array
    # input : pillow_img = PIL instance
    #       : target_size = resized width / height
    # output : uint8 numpy array
    """
    resized = pillow_img.resize((target_size, target_size), Image.LINEAR)
    return np.asarray(resized, np.uint8)


