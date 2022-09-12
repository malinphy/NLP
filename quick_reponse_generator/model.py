import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import *
import tensorflow_hub as hub 
def model():
    use_hub = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


    anc_inp = Input(shape =(), dtype = tf.string, name = 'anchor_input')
    pos_inp = Input(shape =(), dtype = tf.string, name = 'positive_input')
    neg_inp = Input(shape =(), dtype = tf.string, name = 'negative_input')

    use_emb = hub.KerasLayer(use_hub, trainable =True, name = 'sentence_encoder')

    anc_emb = use_emb(anc_inp)
    pos_emb = use_emb(pos_inp)
    neg_emb = use_emb(neg_inp)

    # d1_anc = Dense(256, activation = 'relu')(anc_emb)
    # d1_pos = Dense(256, activation = 'relu')(pos_emb)
    # d1_neg = Dense(256, activation = 'relu')(neg_emb)

    final = tf.keras.layers.Concatenate(axis=-1)([anc_emb, pos_emb, neg_emb])
    final = Dropout(0.2)(final)

    return Model(inputs = [anc_inp, pos_inp, neg_inp], outputs = final)