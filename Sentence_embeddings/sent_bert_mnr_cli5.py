
!pip install --quiet datasets
import pandas as pd 
import numpy as np 
# from google.colab import drive
# drive.mount('/content/drive')
import json
import os 
import gzip
import datasets 

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import *
import tensorflow_hub as hub 

from sklearn.metrics import confusion_matrix

eli5 = datasets.load_dataset('eli5', split = 'train_eli5')

use_hub = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") ## universal sentence encoder model

# title 
# selftext
# answer
#
eli5

print('title length :', len(eli5['title']))
print('selftext length :', len(eli5['selftext']))
print('answers length :', len(eli5['answers']))

df = pd.DataFrame({'title':eli5['title'], 'selftext':eli5['selftext'], 'answer':eli5['answers']})
### as you can see from the data, dataset is corrupted, to eliminate
df.head(4)

answer_len = []
first_answer = []
for i in range(len(df)):
    answer_len.append(len(df['answer'][i]['text']))
    first_answer.append(df['answer'][i]['text'][0])

df['first_answer'] = first_answer

unique_answer = df['first_answer'].unique()
num_unique_answer = len(unique_answer)

unique_questions = df['title'].unique()
num_unique_questions =  len(unique_questions)

print('number of unique answers',num_unique_answer)
print('number of unique questions',num_unique_questions)

neg_pos = []
neg_title = []
neg_answer = []
for i in range(len(df)):
    x = np.random.randint(0, len(df))
    neg_pos.append(x)
    neg_title.append(df['title'][x])
    neg_answer.append(df['first_answer'][x])

df['neg_title'] = neg_title
df['neg_answer'] = neg_answer


def distance_calc(y_true, y_pred):
    anchor, positive, negative = tf.split(y_pred, 3, axis=1)
    ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
    an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
    loss = ap_distance - an_distance
    margin = 0
    loss = tf.maximum(loss + margin, 0.0)
    return loss

title_batch = tf.data.Dataset.from_tensor_slices(df['title']).batch(32)
pos_batch = tf.data.Dataset.from_tensor_slices(df['first_answer']).batch(32)
neg_batch = tf.data.Dataset.from_tensor_slices(df['neg_answer']).batch(32)

first_anc_emb = []
first_pos_emb = []
first_neg_emb = []

for i in title_batch:
    first_anc_emb.append(np.array(use_hub(i)))

for i in pos_batch:
    first_pos_emb.append(np.array(use_hub(i)))

for i in neg_batch:
    first_neg_emb.append(np.array(use_hub(i)))

first_anc_emb = np.concatenate(np.array(first_anc_emb), axis = 0)
first_pos_emb = np.concatenate(np.array(first_pos_emb), axis = 0)
first_neg_emb = np.concatenate(np.array(first_neg_emb), axis = 0)

anc_inp = Input(shape =(), dtype = tf.string, name = 'anchor_input')
pos_inp = Input(shape =(), dtype = tf.string, name = 'positive_input')
neg_inp = Input(shape =(), dtype = tf.string, name = 'negative_input')

use_emb = hub.KerasLayer(use_hub, trainable =True)

anc_emb = use_emb(anc_inp)
pos_emb = use_emb(pos_inp)
neg_emb = use_emb(neg_inp)

# d1_anc = Dense(256, activation = 'relu')(anc_emb)
# d1_pos = Dense(256, activation = 'relu')(pos_emb)
# d1_neg = Dense(256, activation = 'relu')(neg_emb)

final = tf.keras.layers.Concatenate(axis=-1)([anc_emb, pos_emb, neg_emb])
final = Dropout(0.2)(final)

triplet_model = Model(inputs = [anc_inp, pos_inp, neg_inp], outputs = final)

triplet_model.compile(
    optimizer = 'Adam',
    loss = distance_calc
)
y_dummy = np.ones(len(df)).reshape(-1,1)
triplet_model.fit([np.array(df['title']),
                   np.array(df['first_answer']),
                   np.array(df['neg_answer'])
                   ],
                   y_dummy,
                   epochs = 1,
                  batch_size = 32*8
                  )


second_anc_emb = []
second_pos_emb = []
second_neg_emb = []

for i in title_batch:
    second_anc_emb.append(np.array(use_hub(i)))

for i in pos_batch:
    second_pos_emb.append(np.array(use_hub(i)))

for i in neg_batch:
    second_neg_emb.append(np.array(use_hub(i)))

second_anc_emb = np.concatenate(np.array(second_anc_emb), axis = 0)
second_pos_emb = np.concatenate(np.array(second_pos_emb), axis = 0)
second_neg_emb = np.concatenate(np.array(second_neg_emb), axis = 0)

first = tf.keras.layers.Dot(axes = 1, normalize = True)([first_anc_emb,first_pos_emb])
second = tf.keras.layers.Dot(axes = 1, normalize = True)([second_anc_emb,second_pos_emb])

first_neg = tf.keras.layers.Dot(axes = 1, normalize = True)([first_anc_emb,first_neg_emb])
second_neg = tf.keras.layers.Dot(axes = 1, normalize = True)([second_anc_emb,second_neg_emb])

for i in range(10):
    print('ANCHOR:',float(first[i]),'--POS:',float(second[i]),'--NEG:',float(second_neg[i]),'--FIRST NEG',float(first_neg[i]))

    
"""# ##### inference 
# Let's assume that we have a dataset that was pictured like below
## our main purpose is embedding the question and finding the  closest answer
## embedding. To do that, I will encode the question and each possible answer.
## both question embeddings and each answer embeddings will be performed
## dot product to find the cos angle in between, question and each answers.
## after that closest answer will have highest cosine similarity.


| Index      | Questions   | Answers |
| :---       |    :----:   |    ---: |
| 1          | q$_1$       | a$_1$   |
| 2          | q$_2$       | a$_2$   |
| 3          | q$_3$       | a$_3$   |
| 4          | q$_4$       | a$_4$   |
"""    

q_0 = []
for i in range(len(second_anc_emb)):
    x = np.array(second_anc_emb[0]).reshape(1,512)
    y = np.array(second_pos_emb[i]).reshape(1,512)
    q_0.append(tf.keras.layers.Dot(axes = 1, normalize = True)([x,y]))
    
print(tf.math.top_k(tf.squeeze(tf.squeeze(q_0, axis = -1), axis=-1), k =10)[1])
# second_anc_emb[0]
