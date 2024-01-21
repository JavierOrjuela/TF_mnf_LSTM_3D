import numpy as np
#import seaborn as sns
#import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
import embedded_reber_grammar as erg
import sys
sys.path.insert(1,'/home/javier/DESY_new_repo/trans_finder/src/trans_finder/cta')
from lstm_ed.recurrent_mnf import LSTMCellFlipout as LSTMCellFlipout_tf
from lstm_ed.recurrent import LSTMCellReparameterization as LSTMCellReparameterization
from lstm_ed.recurrent_mnf import LSTMCellMNF as LSTMCellMNF
from lstm_ed.regularizers import NormalKLDivergence as NormalKLDivergence
from lstm_ed.initializers import trainable_glorot_normal_shift


def plot_hist(hist):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['val_accuracy'], label='val_accuracy')
    plt.plot(hist.history['accuracy'], label='train_accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.plot(hist.history['loss'], label='train_loss')
    plt.legend()
    plt.show()

x, y = [], []
n = 3000
for i in range(n):
    x.append(np.asarray(erg.encode_string(erg.generate_valid_string(erg.embedded_rg))))
    y.append(1)
for i in range(n):
    x.append(np.asarray(erg.encode_string(erg.generate_invalid_string(erg.embedded_rg))))
    y.append(0)
x = sequence.pad_sequences(x)
x_train, x_test, y_train, y_test = train_test_split(np.asarray(x), np.asarray(y))
len_data=x_train.shape[0]*x_train.shape[1]
print(f"Number of training samples: {x_train.shape[0]}")
print(f"Number of test samples: {x_test.shape[0]} \n")

sequence_length = x_train.shape[1]
num_chars = x_train.shape[2]
print(f"Length of sequences: {sequence_length}")
print(f"Number of characters: {num_chars}")
batch_size = 256
epochs = 1100

lstm=tf.keras.layers.LSTMCell(
            128, 
            #kernel_initializer=trainable_glorot_normal_shift(mean_init=-3.,std_init=0.001),
           # kernel_regularizer=NormalKLDivergence(scale_factor=1./x_train.shape[0]),
           # recurrent_regularizer=NormalKLDivergence(scale_factor=1./x_train.shape[0]),
           # implementation=2
            )
modeltf = tf.keras.Sequential()
modeltf.add(tf.keras.layers.InputLayer(input_shape=(sequence_length, num_chars)))
modeltf.add(tf.keras.layers.RNN(lstm))
modeltf.add(tf.keras.layers.Dense(32))
modeltf.add(tf.keras.layers.Dense(1, activation='sigmoid'))
##import pdb;pdb.set_trace()
def loss_function(x,y):
    entropy= tf.keras.losses.BinaryCrossentropy(
    from_logits=False)
    loss=entropy(x,y)
    #kl=lstm.kl_div_recurrent_kernel()+lstm.kl_div_kernel()
    return loss#+kl/len_data

modeltf.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])
modeltf.summary()
hist = modeltf.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=1)
import pandas as pd
df=pd.DataFrame(hist.history.items())  
df.to_csv('history_lstm.csv')
