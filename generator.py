from PIL import Image
from preprocess import imagenet_data_augmentation, validation_image_load
import numpy as np
from joblib import Parallel, delayed

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


def _parallel_wrap(meta_item, size, is_train):
    """
    Parallel wrap for multiprocessing
    """
    with Image.open(meta_item[0]) as img:
        if is_train:
            X_item = imagenet_data_augmentation(img, size)
        else:
            X_item = validation_image_load(img, size)
    y_item = np.zeros((1000), np.float32)
    y_item[meta_item[1]] = 1.0
    return [X_item, y_item]

def imagenet_generator_multi(meta_data, size, is_train, batch_size, preprocess_func):
    """
    ImageNet data generator using multiprocessing
    # Inputs
      meta_data = [[path, class_idx], [...]] format meta data
      size = output size of image (integer)
      is_train = True if train, false if validation. True = enable shuffle & data augmentation
      batch_size = batch size of generator
      preprocess_func = color rescaling function of corresponding networks
    # Outputs
      yielding (X_batch, y_batch)
    """
    while True:
        if is_train:
            indices = np.random.permutation(len(meta_data))
        else:
            indices = np.arange(len(meta_data))
        for batch_idx in range(indices.shape[0]//batch_size):
            current_indices = indices[batch_idx*batch_size:(batch_idx+1)*batch_size]
            parallel_result = Parallel(n_jobs=4)( [delayed(parallel_wrap)(meta_data[i], size, is_train)
                                                   for i in current_indices] )
            X_batch = np.asarray([parallel_result[i][0] for i in range(batch_size)], np.uint8)
            y_batch = np.asarray([parallel_result[i][1] for i in range(batch_size)], np.float32)
            yield (preprocess_func(X_batch), y_batch)

def fake_data_generator(size, batch_size, preprocess_func):
    """
    Generating random noise image, label (for benchmarkling)
    """
    while True:
        X_batch = np.random.randint(low=0, high=256, size=(batch_size, size, size, 3), dtype=np.uint8)
        y_batch = np.identity(1000, dtype=np.float32)[np.random.randint(low=0, high=1000, size=batch_size)]
        yield (preprocess_func(X_batch), y_batch)