from __future__ import print_function
import numpy as np
import pandas as pd
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks.tensorboard_v1 import TensorBoard
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, roc_curve, auc, classification_report, r2_score
import warnings

warnings.filterwarnings("ignore")

max_features = 309148
maxlen = 100
batch_size = 32

print('Loading data...')

def read_dataset():
    df = pd.read_csv("./data/tain_DL.csv")
    ln=(len(df.columns)-1)
    X = df[df.columns[0:ln]].values
    X = X.reshape(np.shape(X)[0],np.shape(X)[1])
    Y = np.array(df[df.columns[ln]])
    Y = Y.reshape(np.shape(Y)[0])
    return (X,Y)

X, Y = read_dataset()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

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
          epochs=50,
          validation_data=[x_test, y_test])#,
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
