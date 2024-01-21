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

from lstm_ed.recurrent import LSTMCellReparameterization as LSTMCellReparameterization
from lstm_ed.recurrent import LSTMCellFlipout as LSTMCellFlipout
from lstm_ed.recurrent_mnf import LSTMCellMNF as LSTMCellMNF
from lstm_ed.regularizers import NormalKLDivergence as NormalKLDivergence
from lstm_ed.initializers import trainable_glorot_normal_shift
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
import tensorflow_datasets as tfds
max_len = 200
n_words = 10000
dim_embedding = 128
EPOCHS = 20


# def load_data():
#     # Load data.
#     (X_train, Y_train), (X_test, Y_test) = datasets.imdb.load_data(num_words=n_words)
#     # Pad sequences with max_len.
#     X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
#     X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)
#     return (X_train, Y_train), (X_test, Y_test)

dataset, info = tfds.load('imdb_reviews', with_info=True,
                           as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
VOCAB_SIZE = 1000
len_train=len(train_dataset)
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

lstm=LSTMCellMNF(
            64, 
            #kernel_initializer=trainable_glorot_normal_shift(mean_init=-3.,std_init=0.001),
            #kernel_regularizer=NormalKLDivergence(scale_factor=1./len_train),
            #recurrent_regularizer=NormalKLDivergence(scale_factor=1./len_train),
            implementation=2
            )
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.RNN(lstm),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
def loss_function(x,y):
    entropy= tf.keras.losses.BinaryCrossentropy(
    from_logits=True)
    loss=entropy(x,y)
    kl=lstm.kl_div_recurrent_kernel()+lstm.kl_div_kernel()
    return loss+kl/len_train
model.compile(loss=loss_function,#tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-3),
              metrics=['accuracy'])


model.summary()

hist =model.fit(train_dataset, epochs=100,
                    validation_data=test_dataset,
                    validation_steps=30)

import pandas as pd
df=pd.DataFrame(hist.history.items())  
df.to_csv('history_imbd_lstm.csv')



