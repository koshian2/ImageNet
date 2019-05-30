# Image Net Training

# Training Speed
## No multiprocessing
### ResNet50, BatchSize 1280, Enable AutoContrast, NoMultiprocess
50/50 [==============================] - 161s 3s/step - loss: 6.7898 - top1: 0.0108
1000/1000 [==============================] - 6678s 7s/step - loss: 6.6886 - top1: 0.0070 - val_loss: 6.7898 - val_top1: 0.0108
50/50 [==============================] - 130s 3s/step - loss: 5.5986 - top1: 0.0745
1000/1000 [==============================] - 6040s 6s/step - loss: 5.3710 - top1: 0.0731 - val_loss: 5.5986 - val_top1: 0.0745

### SandboxA, BatchSize 1280, Enable AutoContrast, NoMultiprocess
50/50 [==============================] - 133s 3s/step - loss: 4.8394 - top1: 0.1175
1000/1000 [==============================] - 6522s 7s/step - loss: 5.3644 - top1: 0.0755 - val_loss: 4.8394 - val_top1: 0.1175
50/50 [==============================] - 130s 3s/step - loss: 3.8051 - top1: 0.2334
1000/1000 [==============================] - 5722s 6s/step - loss: 3.8748 - top1: 0.2302 - val_loss: 3.8051 - val_top1: 0.2334

### ResNet50, BatchSize 1280, Enable No-AutoContrast, NoMultiprocess
50/50 [==============================] - 161s 3s/step - loss: 6.4965 - top1: 0.0165
1000/1000 [==============================] - 5526s 6s/step - loss: 6.5093 - top1: 0.0129 - val_loss: 6.4965 - val_top1: 0.0165
Set 0.5 to learning rate
Epoch 2/2
50/50 [==============================] - 123s 2s/step - loss: 7.6657 - top1: 0.0241
1000/1000 [==============================] - 5038s 5s/step - loss: 5.0525 - top1: 0.1011 - val_loss: 7.6657 - val_top1: 0.0241

### SandboxA, BatchSize 1280, Enable No-AutoContrast, NoMultiprocess
50/50 [==============================] - 131s 3s/step - loss: 4.7941 - top1: 0.1184
1000/1000 [==============================] - 6173s 6s/step - loss: 5.3702 - top1: 0.0762 - val_loss: 4.7941 - val_top1: 0.1184
Set 0.5 to learning rate
Epoch 2/2
50/50 [==============================] - 129s 3s/step - loss: 3.6528 - top1: 0.2536
1000/1000 [==============================] - 4998s 5s/step - loss: 3.8580 - top1: 0.2327 - val_loss: 3.6528 - val_top1: 0.2536

## Fakedata
### ResNet50, BatchSize 1280, Fakedata
50/50 [==============================] - 266s 5s/step - loss: 6.9118 - top1: 9.6875e-04
1000/1000 [==============================] - 5321s 5s/step - loss: 6.9986 - top1: 0.0010 - val_loss: 6.9118 - val_top1: 9.6875e-04
Set 0.5 to learning rate
Epoch 2/2
50/50 [==============================] - 172s 3s/step - loss: 6.9086 - top1: 0.0010
1000/1000 [==============================] - 3883s 4s/step - loss: 6.9088 - top1: 0.0010 - val_loss: 6.9086 - val_top1: 0.0010

### SandboxA, BatchSize 1280, Fakedata
50/50 [==============================] - 176s 4s/step - loss: 6.9087 - top1: 0.0010
1000/1000 [==============================] - 3673s 4s/step - loss: 6.9131 - top1: 9.6641e-04 - val_loss: 6.9087 - val_top1: 0.0010
Set 0.5 to learning rate
Epoch 2/2
50/50 [==============================] - 149s 3s/step - loss: 6.9084 - top1: 0.0011
1000/1000 [==============================] - 3378s 3s/step - loss: 6.9087 - top1: 0.0010 - val_loss: 6.9084 - val_top1: 0.0011

## Multiprocessing
### ResNet50, BatchSize 1280, Enable AutoContrast, Enable Multiprocess

### SandboxA, BatchSize 1280, Enable AutoContrast, Enable Multiprocess
