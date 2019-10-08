# tensorboard --logdir=logs/
# TensorFlow and tf.keras
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from time import time
import tensorflow as tf
from tensorflow import keras

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder

num_classes = 2
epochs = 1000

def read_dataset():
    df = pd.read_csv("data/tain_DL.csv")
    ln=(len(df.columns)-1)
    X = df[df.columns[0:ln]].values
    Y = df[df.columns[ln]]
    return (X,Y)

X, Y = read_dataset()
X, Y = shuffle(X, Y, random_state=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=415)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

y_train=tf.keras.utils.to_categorical(y_train,num_classes=num_classes, dtype='float32')
y_test= tf.keras.utils.to_categorical(y_test,num_classes=num_classes, dtype='float32')

model = keras.Sequential([
    tf.keras.layers.Conv1D(1, 3, activation=tf.nn.relu, input_shape=(x_train.shape[1:3])),
    tf.keras.layers.Conv1D(64, 3, activation=tf.nn.relu),
    tf.keras.layers.Conv1D(128, 3, activation=tf.nn.relu),
    tf.keras.layers.MaxPool1D(3),
    tf.keras.layers.MaxPool1D(3),
    tf.keras.layers.Conv1D(128, 3, activation=tf.nn.relu),
    tf.keras.layers.Conv1D(128, 3, activation=tf.nn.relu),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

tensorboard = TensorBoard(log_dir='./logs/{}'.format(time()), histogram_freq=10,
    batch_size=32,
    write_graph=True,
    write_grads=False,
    write_images=True,
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None,
    embeddings_data=None,
    update_freq='epoch'
)

model.compile(optimizer='adam',

              loss='binary_crossentropy',
              metrics=['accuracy'])

history=model.fit(x_train, y_train, batch_size=40, epochs=epochs,validation_split=0.25, verbose=1, callbacks=[tensorboard])


from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
y_pred = model.predict(x_test)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.ravel(y_test), np.ravel(y_pred))

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=40)

model.save(os.getcwd()+"\\weights\\Best_Model_using_ft_weights_CNN.h5")

print(test_acc,test_loss)
