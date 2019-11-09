from __future__ import print_function
import numpy as np
import pandas as pd
from time import time
import tensorflow as tf
from tensorflow import keras

from keras.preprocessing import sequence
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, roc_curve, auc, classification_report, r2_score
import warnings

from Bio import SeqIO

warnings.filterwarnings("ignore")

max_features = 309148
maxlen = 100
batch_size = 32

print('Loading data...')

def read_dataset():
    X, Y = [],[]
    with open("./data/positive_training.fasta", "rU") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            X.append(str(record.seq))
            Y.append(1)
            #print(record.seq)
    with open("./data/negative_training.fasta", "rU") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            X.append(str(record.seq))
            Y.append(0)
    return X, Y
    #print(Y)
            #print(record.seq)
    # df = pd.read_csv("./data/training_data.csv")
    # #df = pd.read_csv("./data/tain_DL.csv")
    # ln=(len(df.columns)-1)
    # X = df[df.columns[0:ln]].values
    # X = X.reshape(np.shape(X)[0],np.shape(X)[1])
    # Y = np.array(df[df.columns[ln]])
    # Y = Y.reshape(np.shape(Y)[0])
    # return (X,Y)
#read_dataset()
#exit()
X, Y = read_dataset()

#######
X_train = np.array(X)
y_train = np.array(Y)

X_train = X_train.reshape(len(X_train),1)

amino_acids ='ACDEFGHIKLMNPQRSTVWXY'
embed = []
for i in range(0, len(X_train)):
    length = len(X_train[i][0])
    pos = []
    counter = 0
    st = X_train[i][0]
    for c in st:
        AMINO_INDEX = amino_acids.index(c)
        pos.append(AMINO_INDEX)
        counter += 1
    while(counter < 800):
        pos.append(21)
        counter += 1
    embed.append(pos)
embed = np.array(embed)

data,Label = shuffle(embed,y_train, random_state=2)

X_train = data
y_train = Label

class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weight_dict = dict(enumerate(class_weight))

print(data, Label)
exit()
######

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = keras.Sequential([
    tf.keras.layers.Embedding(max_features, 128, input_length=maxlen),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

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
          epochs=100,
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

model.save("BDLSTM.h5")

print('Test accuracy :',test_acc,'Test Loss :',test_loss)
print('matthews correlation coefficient ',matthews_corrcoef(y_test, np.ravel(y_pred.round())))
print(classification_report(y_test, np.ravel(y_pred.round()), target_names=['class 1','class 2']))
print('r2 score ',r2_score(y_test, np.ravel(y_pred.round())))
