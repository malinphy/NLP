import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import Model,Input,layers
from tensorflow.keras.layers import *

import tensorflow_hub as hub 
import tensorflow_text as text 

def ner_model(num_tags):
    encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    bert_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2"
#     bert_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2"


    preprocessor = hub.KerasLayer(encoder_url)
    encoder = hub.KerasLayer(bert_url,trainable=True)
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    encoder_inputs = preprocessor(text_input)
    outputs = encoder(encoder_inputs)
    sequence_output = outputs["sequence_output"]
    sequence_output = Dropout(0.3)(sequence_output)
    final_layer = Dense(num_tags, activation = 'softmax')(sequence_output)
    return Model(inputs = text_input, outputs = final_layer)
