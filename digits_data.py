import numpy as np
import warnings
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import Model, Sequential

mnist = tf.keras.datasets.mnist
train, valid = mnist.load_data()

import matplotlib.pyplot as plt
plt.ion()



plt.imshow(train[0][0])

plt.figure()
plt.imshow((train[0][0] + train[0][1])/2)



tx = train[0]
ty = train[1]
vx = valid[0]
vy = valid[1]

fives = tx[ty==5]
plt.imshow((fives[0] + fives[1])/2)
plt.close('all')

idx1 = 0
idx2 = 1

indices = [idx1, idx2]

if len(set([ty[i] for i in indices])) < len(indices):
    raise ValueError(f"Labels must all be unique, {indices}")

new_im = np.mean(tx[indices], dtype=np.uint8, axis=0)

new_label = np.zeros(10, dtype=np.uint8)
new_label[[ty[i] for i in indices]] = 1

def combine_images(indices, tx, ty):
    if len(set([ty[i] for i in indices])) < len(indices):
        raise ValueError(f"Labels must all be unique, {indices}")
    return np.mean(tx[indices], dtype=np.uint16, axis=0)


class BadTime(Exception):
    pass





def make_data(ndata, tx, ty, max_iter=1000):
    all_idx = np.arange(len(tx),dtype=np.uint16)
    ims = np.zeros((ndata,28,28), dtype=np.uint16)
    labels = np.zeros((ndata,10), dtype=np.uint8)
    i = 0
    choices = set()
    choice = np.random.choice(all_idx, size=2, replace=True)
    try:
        while i < ndata:
            j = 0
            choice = tuple(choice)
            while choice in choices:
                choice = tuple(np.random.choice(all_idx, size=2, replace=True))
                j += 1
                if j >= max_iter:
                    raise BadTime
            choices.add(choice)
            choice = list(choice)
            try:
                ims[i] = combine_images(choice, tx, ty)
            except ValueError:
                j += 1
                continue
            labels[i,[ty[i] for i in choice]] = 1
            i += 1
            j += 1
            if i % (ndata//10) == 0:
                print('.', end='')
                sys.stdout.flush()
        print('')
    except BadTime:
        warnings.warn(f'Could not generate enough data.  Wanted {ndata}, got {i}')
        ims = ims[:i]
        labels = labels[:i]
    return np.array(ims, dtype=np.uint8), labels


t_im, t_lab = make_data(2**16, tx, ty)
v_im, v_lab = make_data(2**13, vx, vy)

np.savez_compressed('digital_data.npz', t_im, v_im, t_lab, v_lab)

sgd = tf.optimizers.SGD(0.01)

model = Sequential()
model.add(Dense(10, input_shape=(28*28,)))

model.compile(loss='binary_crossentropy',
            optimizer=sgd, metrics=['accuracy'])

hist = model.fit(t_im.reshape((-1,784)),t_lab, epochs=10,
        validation_data=[v_im.reshape((-1,784)),v_lab])
