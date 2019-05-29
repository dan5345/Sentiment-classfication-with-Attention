import re 
import time
import gc
import random
import os

import numpy as np
import pandas as pd

from tqdm import tqdm 

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

embed_size = 300
max_features = 120000
maxLen = 72

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)


def load_and_prec():
    train_df = pd.read_csv('train.csv')
    dev_df = pd.read_csv('dev.csv')
    test_df = pd.read_csv('test.csv')
    print("Train shape: ",train_df.shape)
    print("Test shape: ", test_df.shape)
    print('Dev shape: ', dev_df.shape)
    
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: x.lower())
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: x.lower())
    dev_df["question_text"] = dev_df["question_text"].progress_apply(lambda x: x.lower())
    
    # Clean the text
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_text(x))
    dev_df["question_text"] = dev_df["question_text"].progress_apply(lambda x: clean_text(x))

    
    # Clean numbers
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_numbers(x))
    dev_df["question_text"] = dev_df["question_text"].progress_apply(lambda x: clean_numbers(x))

    
    # Clean speelings
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
    test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
    dev_df["question_text"] = dev_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))

    
    
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values
    dev_X = dev_df["question_text"].fillna("_##_").values
    
    tokenizer = Tokenizer(num_words = max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)
    dev_X = tokenizer.texts_to_sequences(dev_X)
    
    train_X = pad_sequences(train_X, maxlen = maxLen)
    test_X = pad_sequences(test_X, maxlen = maxLen)
    dev_X = pad_sequences(dev_X, maxlen = maxLen)
    
    train_y = train_df['target'].values
    test_y = test_df['target'].values
    dev_y = dev_df['target'].values
    
    np.random.seed(SEED)
    trn_idx = np.random.permutation(len(train_X))
    
    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    
    
    return train_X,dev_X, test_X, train_y,dev_y,test_y, tokenizer.word_index

def load_glove(word_index):
    EMBEDDING_FILE = 'glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 

if __name__ == '__main__':
    tqdm.pandas()

    start_time = time.time()

    train_X,dev_X, test_X, train_y,dev_y,test_y, word_index = load_and_prec()
    embedding_matrix = load_glove(word_index)

    total_time = (time.time() - start_time) / 60
    print("Took {:.2f} minutes".format(total_time))
    np.save("train_X.npy", train_X)
    np.save("train_y.npy", train_y)
    np.save("dev_X.npy", dev_X)
    np.save("dev_y.npy", dev_y)
    np.save("test_X.npy", test_X)
    np.save("test_y.npy", test_y)
    np.save("embed.npy",embedding_matrix)