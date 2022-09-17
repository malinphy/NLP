#### snli prediction
from google.colab import drive
drive.mount('/content/drive')
import tensorflow as tf 
from tensorflow import keras 
import tensorflow_hub as hub

# from tensorflow.keras import Model, layers, Input
# from tensorflow.keras.layers import *
from model import model
import pickle
from sklearn.metrics import confusion_matrix,f1_score,classification_report
import pandas as pd
use  = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# test_path = 'drive/MyDrive/Colab Notebooks/datasets/snli/snli_1.0_test.csv'
# selected_columns = ['sentence1','sentence2','label1']
# test_df =  pd.read_csv(test_path,usecols = selected_columns)
# test_df = test_df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis", "label1": "label"})

p1 = 'This church choir sings to the masses as they sing joyous songs from the book at a church.'
h1 = 'The church has cracks in the ceiling.'

def prediction(premise,hypothesis):
    pkl_file = open('LE.pkl', 'rb')
    LE = pickle.load(pkl_file) 
    pkl_file.close()
    new_df = pd.DataFrame({'p':[premise],'h':[premise]}) 
    
    use_model = model()
    use_model.load_weights('drive/MyDrive/Colab Notebooks/snli/USE_snli_weights.h5')

    return str(LE.inverse_transform(tf.math.top_k(use_model.predict([new_df['p'], new_df['h']]))[1]))

print(prediction(p1,h1))

