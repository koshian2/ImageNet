import pickle
import os
import glob
from tqdm import tqdm

def load_caches(image_directory_path):
    """
    Load cache file of image lists
    # inputs : image directroy root path (root/train, root/test)
    # outputs: image metadata
    """

    def enumerate_classes(directory):
        classes = []
        for subdir in sorted(glob.glob(os.path.join(directory, "*"))):
            if os.path.isdir(subdir):
                classes.append(subdir.replace("\\", "/"))
        return classes

    # simple version
    def enumerate_files(directory, class_idx):
        files = []
        for file in sorted(glob.glob(os.path.join(directory, "*"))):
            if os.path.isfile(file):
                files.append([file.replace("\\", "/"), class_idx])
        return files

    def list_all_images(split):
        lists = []
        for i, dir in tqdm(enumerate(enumerate_classes(os.path.join(image_directory_path, split)))):
            lists += enumerate_files(dir, i)
        return lists

    cache_path = "cache.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fp:
            return pickle.load(fp)
    else:
        result = {}
        result["train"] = list_all_images("train")
        result["val"] = list_all_images("val")
        with open(cache_path, "wb") as fp:
            pickle.dump(result, fp)
        return result
