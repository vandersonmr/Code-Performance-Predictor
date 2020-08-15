"""
This example shows how to perform graph classification with a synthetic dataset
of Delaunay triangulations, using a graph attention network (Velickovic et al.)
in batch mode.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

from spektral.datasets import delaunay
from spektral.layers import *
from spektral.utils.convolution import localpooling_filter
import gc

# Load data
#A, X, y = delaunay.generate_data(return_type='numpy', classes=[0, 5])
print("Loading dataset")
A_train, A_test, \
x_train, x_test, \
y_train, y_test = (0,0,0,0,0,0)
with np.load("small150.npz", allow_pickle=True) as alldata:
    A, X, y = alldata["arr_0"]

    A = np.swapaxes(A,0,1)
    X = np.swapaxes(X,0,1)
    y_c = np.zeros(shape=(y.shape[0],5))

    for i in range(0, y.shape[0]):
        if y[i] > 1.3: 
            y_c[i,3]=1
        elif y[i] < 0.45:
            y_c[i,0]=1
        elif (y[i] <= 0.8 and y[i] >= 0.45):
            y_c[i,1]=1
        elif (y[i] <= 1.33 and y[i] > 0.8):
            y_c[i,2]= 1
        y_c[i,4] = y[i]
    y = y_c
    print(np.sum(y_c,axis=0))
    
    idx = np.argwhere(np.all(X[..., :] == 0, axis=(0,1,2)))
    X = np.delete(X, idx, axis=3)

    print(A.shape)
    print(X.shape)
    print(y.shape)
    
    A[:,0,:,:] = GraphConv.preprocess(A[:,0,:,:]).astype('f4')
    A[:,1,:,:] = GraphConv.preprocess(A[:,1,:,:]).astype('f4')
    
    A_train, A_test, \
    x_train, x_test, \
    y_train, y_test = train_test_split(A, X, y, test_size=0.1)
K = 2 # propagation
gc.collect()
# Parameters
N = X.shape[-2]          # Number of nodes in the graphs
F = X.shape[-1]          # Original feature dimensionality
n_classes = y_c.shape[-1]-1  # Number of classes
learning_rate =  0.000005     # Learning rate for Adam
epochs = 2000         # Number of training epochs
batch_size = 128          # Batch size
iterations = 1          # Number of iterations to approximate each ARMA(1)
order = 2               # Order of the ARMA filter (number of parallel stacks)
share_weights = True    # Share weights in each ARMA stack
dropout = 0.5           # Dropout rate applied between layers
dropout_skip = 0.75     # Dropout rate for the internal skip connection of ARMA
l2_reg = 5e-5           # L2 regularization rate
learning_rate = 1e-2    # Learning rate
es_patience = 100       # Patience for early stopping

# Model definition
X1_in = Input(shape=(N, F))
A1_in = Input((N, N))
fltr1 = A_train[:,0,:,:]
fltr2 = A_train[:,1,:,:]

conv =  GraphConv(45,activation='relu',use_bias=False)
mincut = MinCutPool(N // 2)
conv2 =  GraphConv(55,activation='relu',use_bias=False)
pool = GlobalAttnSumPool()

gc2_1 = conv([X1_in, A1_in])
gc2_1, A1 = mincut([gc2_1,A1_in])
gc2_1 = conv2([gc2_1, A1])
pool_1 =  pool(gc2_1)

X2_in = Input(shape=(N, F))
A2_in = Input((N, N))

#fltr2 = ARMAConv.preprocess(A_train[:,1,:,:]).astype('f4')

gc2_2 = conv([X2_in, A2_in])
gc2_2, A2 = mincut([gc2_2,A2_in])
gc2_2 = conv2([gc2_2, A2])
pool_2 = pool(gc2_2)


d1 = Dense(200,activation='relu')(pool_1)
d2 = Dense(200,activation='relu')(pool_2)

merged = Concatenate()([d1, d2])

merged1 = Dense(800,activation='relu')(merged)
merged2 = Dense(32,activation='relu')(merged1)

classe = Dense(n_classes, name="class",activation="softmax")(merged2)
speedup = Dense(1, name="speddup")(merged2)

# Build model
model = Model(inputs=[X1_in, A1_in,X2_in, A2_in], outputs=[classe,speedup])
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss=['categorical_crossentropy', 'mse'],loss_weights=[1, 0.00005], weighted_metrics=['acc'])
model.summary()

# Train model
history = model.fit([x_train[:,0,:,:],fltr1, x_train[:,1,:,:], fltr2],
        [y_train[:,0:4],y_train[:,4]],
          batch_size=batch_size,
          validation_split=0.05,
          epochs=epochs)

plt.plot(np.array(history.history['class_acc']))
plt.plot(np.array(history.history['val_class_acc']))
plt.title('model accuracy')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['treino', 'val'], loc='upper left')
plt.savefig("test.pdf")

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([x_test[:,0,:,:], A_test[:,0,:,:], x_test[:,1,:,:], A_test[:,1,:,:]],
        [y_test[:,0:4],y_test[:,4]],
                              batch_size=batch_size)
print('Done. Test loss: {:.4f}. Test acc: {:.2f}'.format(*eval_results))

y_pred = model.predict([x_test[:,0,:,:], A_test[:,0,:,:], x_test[:,1,:,:], A_test[:,1,:,:]])
y_pred = np.argmax(y_pred[0], axis=-1)
y_test = np.argmax(y_test[:,0:4], axis=-1)

print(y_pred)
print(y_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
