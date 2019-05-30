# Image Net Training

# Training Speed
* ResNet50, BatchSize 1280, Enable AutoContrast, NoMultiprocess
50/50 [==============================] - 161s 3s/step - loss: 6.7898 - top1: 0.0108
1000/1000 [==============================] - 6678s 7s/step - loss: 6.6886 - top1: 0.0070 - val_loss: 6.7898 - val_top1: 0.0108
50/50 [==============================] - 130s 3s/step - loss: 5.5986 - top1: 0.0745
1000/1000 [==============================] - 6040s 6s/step - loss: 5.3710 - top1: 0.0731 - val_loss: 5.5986 - val_top1: 0.0745

* SandboxA, BatchSize 1280, Enable AutoContrast, NoMultiprocess
50/50 [==============================] - 133s 3s/step - loss: 4.8394 - top1: 0.1175
1000/1000 [==============================] - 6522s 7s/step - loss: 5.3644 - top1: 0.0755 - val_loss: 4.8394 - val_top1: 0.1175
50/50 [==============================] - 130s 3s/step - loss: 3.8051 - top1: 0.2334
1000/1000 [==============================] - 5722s 6s/step - loss: 3.8748 - top1: 0.2302 - val_loss: 3.8051 - val_top1: 0.2334

* ResNet50, BatchSize 1280, Enable No-AutoContrast, NoMultiprocess
50/50 [==============================] - 161s 3s/step - loss: 6.4965 - top1: 0.0165
1000/1000 [==============================] - 5526s 6s/step - loss: 6.5093 - top1: 0.0129 - val_loss: 6.4965 - val_top1: 0.0165
Set 0.5 to learning rate
Epoch 2/2
50/50 [==============================] - 123s 2s/step - loss: 7.6657 - top1: 0.0241
1000/1000 [==============================] - 5038s 5s/step - loss: 5.0525 - top1: 0.1011 - val_loss: 7.6657 - val_top1: 0.0241

* SandboxA, BatchSize 1280, Enable No-AutoContrast, NoMultiprocess
50/50 [==============================] - 131s 3s/step - loss: 4.7941 - top1: 0.1184
1000/1000 [==============================] - 6173s 6s/step - loss: 5.3702 - top1: 0.0762 - val_loss: 4.7941 - val_top1: 0.1184
Set 0.5 to learning rate
Epoch 2/2
50/50 [==============================] - 129s 3s/step - loss: 3.6528 - top1: 0.2536
1000/1000 [==============================] - 4998s 5s/step - loss: 3.8580 - top1: 0.2327 - val_loss: 3.6528 - val_top1: 0.2536