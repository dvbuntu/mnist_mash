import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import Model, Sequential
import sklearn.metrics


import matplotlib.pyplot as plt
plt.ion()
data = np.load('digital_data.npz')

t_im = data['arr_0']
v_im = data['arr_1']
t_lab = np.array(data['arr_2'], dtype=np.float32)
v_lab = np.array(data['arr_3'], dtype=np.float32)

data.close()

K = tf.keras.backend

# jaccard index loss function
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def sk_acc(y_true, y_pred):
    L = y_true.shape[0]
    pt = tf.cast(y_pred > .5, dtype=y_true.dtype)
    correct = tf.cast(K.sum(y_true*pt,axis=-1) == 2, 'int32')
    foo = tf.cast((1-y_true)*(1-pt), 'int32')
    correct *= tf.cast(K.sum(foo,axis=-1) == 8, 'int32')
    return K.sum(correct)/L

def sk_jacc(y_true, y_pred):
    return sklearn.metrics.jaccard_similarity_score(y_true, y_pred > .5)


# dopey ML model
sgd = tf.optimizers.Adam(0.01)
model = Sequential()
model.add(Dense(10, input_shape=(28*28,),
          activation='sigmoid'))

#model.add(Dense(10, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer=sgd, metrics=['accuracy',
				])

hist = model.fit(t_im.reshape((-1,784)),t_lab, epochs=10,
        validation_data=[v_im.reshape((-1,784)),v_lab])

pred = model.predict(v_im.reshape((-1,784)))

# % exact matches for all labels in an image
print(sklearn.metrics.accuracy_score(v_lab, pred > .5))

# intersection / union of sets (length, really)
print(sklearn.metrics.jaccard_similarity_score(v_lab, pred > .5))

# % true labels correctly identified
print(np.sum(v_lab * pred)/len(v_im)/2)

# look at averages of mnist data
mnist = tf.keras.datasets.mnist
train, valid = mnist.load_data()

mt_im = train[0]
mt_lab = train[1]

plt.matshow(np.mean(mt_im[mt_lab == 0],axis=0))

fig, axes = plt.subplots(4,3)
M = np.zeros((10,28,28), dtype=np.float32)
for i in range(10):
    M[i] = np.mean(mt_im[mt_lab == i],axis=0)
    x = i // 3
    y = i % 3
    axes[x,y].matshow(M[i])

ndata = len(t_im)
A = np.zeros((ndata,10,10), dtype=np.float32)
V = np.zeros((len(v_im),10,10), dtype=np.float32)
for i in range(10):
    for j in range(i+1, 10):
        A[:,i,j] = np.mean((t_im - M[i]) - M[j], axis=(1,2))
        A[:,j,i] = A[:,i,j]
        V[:,i,j] = np.mean((v_im - M[i]) - M[j], axis=(1,2))
        V[:,j,i] = V[:,i,j]
    print(i)

plt.matshow(np.mean(A,axis=0))

# Make a simple heuristic
## grab the two labels closest to zero for a given image
A_adj = np.copy(A) * np.tri(10,10, k=-1)
A_adj[A_adj == 0] = np.max(A_adj)
A_adj = np.abs(A_adj)
mins = np.argmin(A_adj.reshape((len(A_adj),-1)),axis=1)
min_idx = np.zeros((len(t_im), 2))
min_idx[:,0] = mins//10
min_idx[:,1] = mins % 10

pred = np.zeros((len(t_im), 10))

# fine, do it slow
import tqdm
for i in tqdm.tqdm(range(len(pred))):
    pred[i,np.array(min_idx[i], dtype=np.uint8)] = 1

print(sklearn.metrics.accuracy_score(t_lab, pred))
print(np.sum(t_lab * pred)/len(t_im)/2)
