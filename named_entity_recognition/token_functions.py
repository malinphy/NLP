#### bert ner token functions
import tensorflow_hub as hub 
import tokenizers
from tokenizers import BertWordPieceTokenizer
import tensorflow as tf 
from tensorflow.keras.preprocessing.sequence import pad_sequences

encoder_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
bert_url = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2'
bert_layer = hub.KerasLayer(bert_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy().decode("utf-8")
tokenizer = BertWordPieceTokenizer(vocab=vocab_file, lowercase=False)



def token_aligner(sentences_train,tags_train):
    var2 = []
    for j in range(len(sentences_train)):
        encoded_sentence = tokenizer.encode(sentences_train[j])
        tags = tags_train[j].split(' ')
        counter = 0
        var1 = []
        for i in range(1, len(encoded_sentence.offsets)- 1):
            x = encoded_sentence.offsets[i][0]
            y = encoded_sentence.offsets[i-1][1]

            if x!=y :
                counter +=1
                var1.append(tags[counter])
            if x == y:
                var1.append(tags[counter]) 
        var2.append(var1)
    
    return var2

def encoder(x):
    tokenizer = BertWordPieceTokenizer(vocab=vocab_file, lowercase=True)
    ids = []
    type_ids = []
    attention_mask = []
    ids_len = []
    tokens= []
    for i in range(len(x)):
        var1 = tokenizer.encode(x[i])
        ids.append(var1.ids)
        type_ids.append(var1.type_ids)
        attention_mask.append(var1.attention_mask)
        tokens = var1.tokens
        ids_len.append(len(var1.ids))
    return ids, type_ids, attention_mask,tokens,ids_len


def tag_encoder_2(tags,tags_2enc):
    '''
    encoding tags using tag corpus(enc_2tags)
    '''

    encoded_tags = []

    for i in tags:
        t1 = []

        for j in i:

            t1.append(tags_2enc[str(j)])
        encoded_tags.append(t1)

    return encoded_tags
    
def padder(x,pad_len):
    padded_var = pad_sequences(
    x,
    maxlen=pad_len,
    dtype='int32',
    padding='post',
    truncating='post',
    value=0.0
    )
    return padded_var