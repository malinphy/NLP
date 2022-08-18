# -*- coding: utf-8 -*-


import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras 
!pip install --quiet datasets
import datasets
import tensorflow_hub as hub
from tensorflow.keras import Model, layers, Input
from tensorflow.keras.layers import *

import matplotlib.pyplot as plt
import seaborn as sns 

use  = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") ## universal sentence encoder model

train_set = datasets.load_dataset('snli', split = 'train')
test_set = datasets.load_dataset('snli', split = 'test')
valid_set = datasets.load_dataset('snli', split = 'validation')

def df_maker(x):

    df = pd.DataFrame({'premise':x['premise'], 'hypothesis':x['hypothesis'], 'label':x['label']})
    # print('unique labels',df['label'].unique())
    minus_labels = np.where(df['label'] == -1)[0]
    df = df.drop(minus_labels).reset_index()
    # print('unique labels',df['label'].unique())
    return df

train_df = df_maker(train_set)
test_df = df_maker(test_set)
validation_df = df_maker(valid_set)

validation_df.head(3)

prod = []
for i in range(10):
    x = use([validation_df['premise'][i]])
    y = use([validation_df['hypothesis'][i]])
    prod.append(np.inner(x,y))

prod

print(len(train_df))
print(len(test_df))
print(len(validation_df))

num_labels = train_df['label'].unique()
print(num_labels)

left_input = Input(shape = (), name = 'left_input', dtype =tf.string)
right_input = Input(shape = (), name = 'right_input', dtype =tf.string)

encoder_layer = hub.KerasLayer(use, trainable = True)

left_encoder = encoder_layer(left_input)
right_encoder = encoder_layer(right_input)
dot_product_layer = tf.keras.layers.Dot(axes = -1, normalize=True)([left_encoder, right_encoder])
dense_layer = Dense(len(num_labels), activation = 'softmax')(dot_product_layer)

use_model = Model(inputs = [left_input, right_input], outputs = dense_layer)

use_model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam()
)

use_model.fit(
    [test_df['premise'], test_df['hypothesis']],
    test_df['label'],
    epochs = 1
)

validation_df

prod2 = []
for i in range(10):
    x = encoder_layer([validation_df['premise'][i]])
    y = encoder_layer([validation_df['hypothesis'][i]])
    prod2.append(np.inner(x,y))

prod2

prod

