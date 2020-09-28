# -*- coding: utf-8 -*-
import torch 
import string 
import argparse
import pandas as pd 
from unidecode import unidecode 
from nltk.corpus import stopwords 
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

def clean(description): 
    s = unidecode(description) 
    s = text_to_word_sequence(s) 
    table = str.maketrans('','', string.punctuation) 
    s = [word.translate(table) for word in s] 
    return s 

def remove_stop_words(tokenized_words, language):
    if language == 'portuguese':
        stop_words_pt = stopwords.words('portuguese')
        text_clean = [word for word in tokenized_words if word not in stop_words_pt]
    else:
        stop_words_es = stopwords.words('spanish')
        text_clean = [word for word in tokenized_words if word not in stop_words_es]
    
    return text_clean

def remove_digits(tokenized_words):
    table = str.maketrans('','', string.digits)
    return [word.translate(table) for word in tokenized_words if len(word.translate(table)) != 0 ]      

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
def preprocess(text, max_len=62):
    text = remove_digits(text)
    text = " ".join(text)
    return tokenizer.encode(text, max_length=max_len, padding='max_length', truncation=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', dest='file', required=True)
    args = parser.parse_args()

    # read data
    data = pd.read_csv(args.file)                                                                                                              

    # Mapping classes to integers
    classes = {i:classe for classe, i in enumerate(data['category'].unique())}
    data['classes_map'] = data['category'].map(classes)

    # Remove special caracters, and punctuation
    data['processed'] = data['title'].apply(lambda x: clean(x))

    # Remove stop words
    pt_idx = data['language'] == 'portuguese'
    es_idx = data['language'] == 'spanish'
    data['processed'][pt_idx] = data['processed'][pt_idx].apply(lambda x: remove_stop_words(x, 'portuguese'))
    data['processed'][es_idx] = data['processed'][es_idx].apply(lambda x: remove_stop_words(x, 'spanish'))

    data['processed'] = data['processed'].apply(lambda x: preprocess(x))

    torch.save(data, '{}_processed.pt'.format(args.file.split('.')[0]))