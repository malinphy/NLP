### model for the snli

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import Model, layers, Input
from tensorflow.keras.layers import *
import tensorflow_hub as hub

use  = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") ## universal sentence encoder model


def model():
    conv_size = 40
    drop_rate = 0.2
    pool_size = 6
    
    left_input = Input(shape = (), name = 'left_input', dtype = tf.string)
    right_input = Input(shape = (), name = 'right_input', dtype = tf.string)

    encoder_layer = hub.KerasLayer(use, trainable = True)

    left_encoder = encoder_layer(left_input)
    right_encoder = encoder_layer(right_input)
    subtract_layer = tf.keras.layers.Subtract()([left_encoder, right_encoder])
    subtract_layer = tf.math.abs(subtract_layer)

    left_encoder_wide = tf.expand_dims(left_encoder,axis=-1)
    right_encoder_wide = tf.expand_dims(right_encoder,axis=-1)
    subtract_wide = tf.expand_dims(subtract_layer,axis=-1)

    left_convo  = Conv1D(512, conv_size)(left_encoder_wide)
    left_pool = MaxPool1D(pool_size)(left_convo)

    right_convo  = Conv1D(512, conv_size)(right_encoder_wide)
    right_pool = MaxPool1D(pool_size)(right_convo)

    sub_convo = Conv1D(512, conv_size)(subtract_wide)
    sub_pool = MaxPool1D(pool_size)(sub_convo)

    left_pool = Flatten()(left_pool)
    right_pool = Flatten()(right_pool)
    sub_pool = Flatten()(sub_pool)


# concat_layer = tf.keras.layers.Concatenate(axis=-1)([left_encoder, right_encoder, subtract_layer])
    concat_layer = tf.keras.layers.Concatenate(axis=-1)([left_pool, right_pool,sub_pool])

# d1_layer = Dense(1024, activation = 'relu')(concat_layer)
# d1_layer = Dropout(drop_rate)(d1_layer)
    d2_layer = Dense(512, activation = 'relu')(concat_layer)
    d2_layer = Dropout(drop_rate)(d2_layer)
    d3_layer = Dense(256, activation = 'relu')(d2_layer)
    d3_layer = Dropout(drop_rate)(d3_layer)
    d3_layer = Dense(64, activation = 'relu')(d2_layer)
    d3_layer = Dropout(drop_rate)(d3_layer)
    d4_layer = Dense(3, activation = 'softmax')(d3_layer)

    return Model(inputs = [left_input, right_input], outputs = d4_layer)
