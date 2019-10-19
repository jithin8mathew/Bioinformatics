from __future__ import print_function
import numpy as np
import pandas as pd
from time import time
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, roc_curve, auc, classification_report, r2_score
import warnings

warnings.filterwarnings("ignore")

max_features = 309148
maxlen = 1000
batch_size = 32

print('Loading data...')

def read_dataset():
    df = pd.read_csv("./data/tain_DL.csv")
    ln=(len(df.columns)-1)
    X = df[df.columns[0:ln]].values
    X = X.reshape(np.shape(X)[0],np.shape(X)[1],1)
    Y = np.array(df[df.columns[ln]])
    Y = Y.reshape(np.shape(Y)[0],1)
    return (X,Y)

X, Y = read_dataset()
X, Y = shuffle(X, Y, random_state=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

y_train = np.array(y_train)
y_test = np.array(y_test)
y_train=tf.keras.utils.to_categorical(y_train,num_classes=2, dtype='float32')
y_test= tf.keras.utils.to_categorical(y_test,num_classes=2, dtype='float32')

input = tf.keras.layers.Input(shape=(x_train.shape[1:3]))
conv1 = tf.keras.layers.Conv1D(1, 3, activation=tf.nn.relu)(input)
conv2 = tf.keras.layers.Conv1D(64, 3, activation=tf.nn.relu)(conv1)
max1 = tf.keras.layers.MaxPool1D()(conv2)
max2 = tf.keras.layers.MaxPool1D()(max1)
conv3 = tf.keras.layers.Conv1D(128, 3, activation=tf.nn.relu)(max2)
conv4 = tf.keras.layers.Conv1D(128, 3, activation=tf.nn.relu)(conv3)
max3 = tf.keras.layers.GlobalMaxPool1D()(conv4)
flat = tf.keras.layers.Flatten()(max3)
flat = tf.keras.layers.Reshape((1, 128),input_shape=(flat.get_shape()))(flat)
lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(flat)
flat2 = tf.keras.layers.Flatten()(lstm)
drop1= tf.keras.layers.Dropout(0.5)(flat2)
out= tf.keras.layers.Dense(2, activation=tf.nn.softmax)(drop1)#(flat2)#(drop1)

model = keras.Model(input, out)
print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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

print('Training...')
history= model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1000,
          validation_data=[x_test, y_test],
          callbacks=[tensorboard])

# Prediction and ROC/ AUC curve plotting
y_pred = model.predict(x_test)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.ravel(y_test), np.ravel(y_pred))
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)

model.save("./weights/BDLSTM.h5")

print('Test accuracy :',test_acc,'Test Loss :',test_loss)
print('matthews correlation coefficient ',matthews_corrcoef(np.ravel(y_test.round()), np.ravel(y_pred.round())))
print(classification_report(np.ravel(y_test.round()), np.ravel(y_pred.round()), target_names=['class 1','class 2']))
print('r2 score ',r2_score(np.ravel(y_test.round()), np.ravel(y_pred.round())))
