import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import Model, Sequential


import matplotlib.pyplot as plt
plt.ion()
data = np.load('digital_data.npz')

t_im = data['arr_0']
v_im = data['arr_1']
t_lab = data['arr_2']
v_lab = data['arr_3']

data.close()




# dopey ML model
sgd = tf.optimizers.SGD(0.01)
model = Sequential()
model.add(Dense(10, input_shape=(28*28,),
          activation='sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer=sgd, metrics=['accuracy'])

hist = model.fit(t_im.reshape((-1,784)),t_lab, epochs=1,
        validation_data=[v_im.reshape((-1,784)),v_lab])

pred = model.predict(v_im.reshape((-1,784)))

print(sklearn.metrics.accuracy_score(v_lab, pred > .5))
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
