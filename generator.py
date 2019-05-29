from PIL import Image
from preprocess import imagenet_data_augmentation, validation_image_load
import numpy as np
import threading

# https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def imagenet_generator(meta_data, size, is_train, batch_size, preprocess_func):
    """
    ImageNet data generator
    # Inputs
      meta_data = [[path, class_idx], [...]] format meta data
      size = output size of image (integer)
      is_train = True if train, false if validation. True = enable shuffle & data augmentation
      batch_size = batch size of generator
      preprocess_func = color rescaling function of corresponding networks
    # Outputs
      yielding (X_batch, y_batch)
    """
    X_cache, y_cache = [], []
    while True:
        if is_train:
            indices = np.random.permutation(len(meta_data))
        else:
            indices = np.arange(len(meta_data))
        for i in indices:
            with Image.open(meta_data[i][0]) as img:
                if is_train:
                    X_item = imagenet_data_augmentation(img, size)
                else:
                    X_item = validation_image_load(img, size)
            y_item = np.zeros((1000), np.float32)
            y_item[meta_data[i][1]] = 1.0
            X_cache.append(X_item)
            y_cache.append(y_item)
            if len(X_cache) == batch_size:
                X_batch = np.asarray(X_cache, np.uint8)
                y_batch = np.asarray(y_cache, np.float32)
                X_cache, y_cache = [], []
                yield (preprocess_func(X_batch), y_batch)