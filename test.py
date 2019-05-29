import os
import glob
from tqdm import tqdm
import time
from PIL import Image, ImageOps

base_path = "E:/Python/ImageNet/imagenet_2"

def hoge():
    train_dir = os.path.join(base_path, "train")
    train_classes = []
    for subdir in sorted(os.listdir(train_dir)):
        if os.path.isdir(os.path.join(train_dir, subdir)):
            train_classes.append(subdir)

    val_dir = os.path.join(base_path, "val")
    val_classes = []
    for subdir in sorted(os.listdir(val_dir)):
        if os.path.isdir(os.path.join(val_dir, subdir)):
            val_classes.append(subdir)

    flags = [x == y for x, y in zip(train_classes, val_classes)]
    print(all(flags))
    print(len(train_classes))
    print(len(val_classes))

def hoge():
    for file in sorted(os.walk(base_path+"/train/n01440764")):
        print(file)

def load_time():
    root_dir_hdd = base_path+"/train/n01440764/*"
    base_path_ssd = "G:/n01440764/*"
    files = sorted(glob.glob(base_path_ssd))
    start_time = time.time()
    for i in tqdm(range(10)):
        for f in files:
            with Image.open(f) as img:
                #img = ImageOps.autocontrast(img, i*0.1)
                img = img.resize((256, 256), Image.LINEAR)
                #img = ImageOps.autocontrast(img.resize((256, 256), Image.LINEAR), i*0.1)

                # HDD
                # LANCZOS -> 17.33s, 9.83s
                # autocon 11.33s, 10.99s
                # LANCZOS+autocon -> 15.02s, 14.99s
                # LINEAR+autocon -> 13.70, 13.07s

                # SSD
                # LINEAR -> 7.42s 7.49s 12分
                # LANCZOS -> 9.31s 9.31s
                # autocon -> 10.82s 11.04s
                # LANCZOS+autocon -> 14.85s 15.07s
                # LINEAR+autocon -> 13.01s 13.01s 22分

    print(time.time()-start_time)

import tensorflow as tf
import numpy as np

def test():
    root_dir_hdd = base_path+"/train/n01440764/*"
    base_path_ssd = "G:/n01440764/*"
    files = sorted(glob.glob(base_path_ssd))
    start_time = time.time()
    for i in tqdm(range(10)):
        for f in files:
            with Image.open(f) as img:
                array = np.expand_dims(np.asarray(img, np.uint8), 0)
                convert = tf.image.resize_bicubic(array, (256, 256))

test()
