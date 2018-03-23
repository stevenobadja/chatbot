# import all dependencies
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from attention_decoder import AttentionDecoder
from nltk.stem import PorterStemmer
from tqdm import tqdm
import string
import pandas as pd

ps = PorterStemmer()

# return all sentences from the 2 list (q&a)
def return_sents(df_col1, df_col2):
    return [sent for sent in df_col1] + [sent for sent in df_col2]

# return unique words from list of sentences
def return_unique_words(all_sents):
    table = str.maketrans({key: None for key in string.punctuation})
    all_words = [words.split() for words in all_sents]
    word_list = [word.lower() for sublist in all_words for word in sublist]
    word_list = [word.translate(table) for word in word_list]

# *** removed stemming to enhance Monty's reponse
#    word_list = [ps.stem(word) for word in word_list]
    return list(set(word_list))

# return a dataframe of words from 2 list, this data frame is used as a form of hash table
# key would be the index and the value would be the word
def df_to_df(df_col1, df_col2):
    all_sent = return_sents(df_col1, df_col2)
    word_list = return_unique_words(all_sent)
    word_list.insert(0, ' ')
    t_df = pd.DataFrame()
    t_df['word'] = word_list
    t_df['idx'] = t_df.index
    return t_df

# return unique words from list of sentences
def return_unique_words_single(sent):
    table = str.maketrans({key: None for key in string.punctuation})
    all_words = sent.split()
    word_list = [word.lower() for word in all_words]
    word_list = [word.translate(table) for word in word_list]

# *** removed stemming to enhance Monty's reponse
#    word_list = [ps.stem(word) for word in word_list]
    return word_list

# function takes the sentences and the hash table of word index and returns the array
# equivalent index of the words in each sentence
def word_to_array(sents, t_df):
    l = []
    l2 = []
    for sent in sents:
        b = []
        a = return_unique_words_single(sent)
        for w in a:
            try:
                b.append(t_df.loc[t_df.word == w, 'idx'].iloc[0])
            except:
                b.append(0)
        l.append(a)
        l2.append(b)
    return l, l2

# decodes the array back into string so humans can understand what Monty is saying
def array_to_string(ar, t_df):
    c = [t_df.loc[t_df.idx == i, 'word'].iloc[0] for i in ar]
    s = ' '.join(c)
    return s

# one hot encode the array sequence
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)

# decode a one hot encoded array sequence
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]

# transform the X & y into one hot format and reshape it into proper input shape
def transform_xy(sequence_in, sequence_out, n_features):
    X = one_hot_encode(sequence_in, n_features)
    y = one_hot_encode(sequence_out, n_features)
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))
    return X,y

# for user interaction purposes Monty breaks down each word from each sentence and looks
# up the equivalent
def sent_to_array(sent, t_df):
    a = []
    b = []
    a = return_unique_words_single(sent)
    for w in a:
        try:
            b.append(t_df.loc[t_df.word == w, 'idx'].iloc[0])
        except:
            b.append(0)
    return a, b

# for user interaction purposes Monty breaks down each word from the user input and encode
# and shapes it into something the model can use to predict
def transform_x(sequence_in, n_features):
    X = one_hot_encode(sequence_in, n_features)
    X = X.reshape((1, X.shape[0], X.shape[1]))
    return X

# for ease of use this function was created to make it easier to interact with Monty
# it takes the user input and returns a response
def get_response(sent, t_df, max_length, n_features, model):
    w, q = sent_to_array(sent, t_df)
    q_pad = pad_sequences([q], maxlen=max_length, padding='post')
    X2 = transform_x(q_pad[0], n_features)
    yhat2 = model.predict(X2, verbose=0)
    return array_to_string(one_hot_decode(yhat2[0]), t_df)
