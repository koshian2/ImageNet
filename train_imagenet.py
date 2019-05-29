import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.contrib.tpu.python.tpu import keras_support
import numpy as np
import os

from utils import load_caches
from generator import imagenet_generator
from imagenet_models import resnet50


def top1(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true, y_pred)

def top5(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

USE_TPU = True
IMAGE_NET_ROOT = "E:/Python/ImageNet/imagenet_2" if not USE_TPU else "./imagenet_2"

class CustomScuduling():
    def __init__(self, epoch_offset, initial_lr):
        self.epoch_offset = epoch_offset
        self.initial_lr = initial_lr

    def scheduler(self, epoch):
        current_epoch = self.epoch_offset + epoch
        x = self.initial_lr
        if current_epoch >= 30: x /= 10.0
        if current_epoch >= 60: x /= 10.0
        if current_epoch >= 90: x /= 10.0
        print("Set", x, "to learning rate")
        return x

class Checkpoint(keras.callbacks.Callback):
    def __init__(self, model, model_name, train_epochs):
        self.modle = model
        self.train_epochs = train_epochs
        self.model_name = model_name
        self.max_val_top1 = 0.0

    def on_epoch_end(self, epoch, logs):
        if logs["val_top1"] > self.max_val_top1:
            self.model.save_weights(self.model_name+".h5", save_format="h5")
            print(f"Val top1 improved {self.max_fal_top1:.04} to {logs['val_top1']:.04}")
            self.max_val_top1 = logs["val_top1"]
        elif epoch-1 == self.train_epochs:
            self.model.save_weights(self.model_name+"_last.h5", save_format="h5")
            print(f"last weights saved")

def train(network, epoch_offset):
    if network == "resnet50":
        model = resnet50.ResNet50(include_top=True, weights=None)
        size = 224
        preprocess = resnet50.preprocess_input

    if USE_TPU:
        train_batch_size, val_batch_size = 1280, 1000
    nb_epoch = 2

    initial_lr = 0.1 * train_batch_size / 256
    model.compile(keras.optimizers.SGD(initial_lr, 0.9), "categorical_crossentropy", 
                  [top1, top5])

    if USE_TPU:
        # convert to tpu model
        tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
        strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
        model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    cache = load_caches(IMAGE_NET_ROOT)
    n_train, n_val = len(cache["train"]), len(cache["val"])

    train_gen = imagenet_generator(cache["train"], size, True, train_batch_size, preprocess)
    val_gen = imagenet_generator(cache["val"], size, False, val_batch_size, preprocess)

    scheduler_obj = CustomScuduling(epoch_offset, initial_lr)
    scheduler = keras.callbacks.LearningRateScheduler(scheduler_obj.scheduler)
    cp = Checkpoint(model, network, nb_epoch)

    model.fit_generator(train_gen, steps_per_epoch=n_train//train_batch_size, epochs=nb_epoch,
                        validation_data=val_gen, validation_steps=n_val//val_batch_size,
                        max_queue_size=3, callbacks=[scheduler])

if __name__ == "__main__":
    train("resnet50", 0)