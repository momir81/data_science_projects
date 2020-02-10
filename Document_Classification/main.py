import h5py
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import json
import os
import argparse
import configparser
from read_data import ReadData
from word_token import WordToken
from text_classify import TextClassify
from word_token import WordToken
from model import Model
import logging
import time
import pickle

logging.basicConfig(level = logging.INFO)

def str2None(v):
    if v == "None":
        return None
    else:
        return v

def save_outputs(file_name, data_name, data):

    h5f = h5py.File(file_name, 'w')

    h5f.create_dataset(data_name, data = data)

    h5f.close()

def read_outputs(file_name, data_name):

    h5f = h5py.File(file_name,'r')

    out = h5f[data_name][:]

    h5f.close()

    return out

def calculate_docs_size(docs):

    tokens = [word for word in word_tokenize(docs)]

    return len(tokens)


if __name__== '__main__':

    '''
    This is the main function call. It process optional arguments from command line and runs whole program.

    Command line arguments:

    --corr corr: in case of calculating most correlated unigrams and bigrams in documents

    --project project: calculates projects of 5 most frequent documents in 2D to see their relationship

    --balance balance: upsample document categories due to higly skewed categories

    '''

    start_time = time.time() 

    parser = argparse.ArgumentParser()

    parser.add_argument('--corr', default = None, type=str2None)  # find correlated unigrams and bigrams in each document

    parser.add_argument('--project', default = None, type=str2None)  # project 5 most frequent documents

    parser.add_argument('--balance', default = None, type=str2None) # upsample classes for learning curves and possible model improvement

    args = parser.parse_args()

    corr = args.corr

    project = args.project

    balance = args.balance

    #read documents from train and test jsonl format and convert them to pandas dataframe

    read_train = ReadData('train.jsonl')

    read_test = ReadData('test.jsonl')

    df_train = read_train.read_data()

    #calculate minimal and maximal number of raw words in documents 

    df_train['doc_size'] = df_train['text'].apply(lambda x:calculate_docs_size(x))

    print('Max len of raw document in train set: {}'.format(df_train['doc_size'].max()))

    print('Min len of raw document in train set: {}'.format(df_train['doc_size'].min()))

    df_test = read_test.read_data()

    text_class = TextClassify(df_train, df_test)

    #encode raw document categories

    y_train = text_class.encode_labels(df_train.category.values)

    y_test = text_class.encode_labels(df_test.category.values)

    #calculate tfidf weights for each word in documents and creates train and test arrays and save them 

    X_train, X_test, tf = text_class.calculate_tfidf()

    X_train_feat, X_test_feat = text_class.get_features(X_train, X_test, y_train)

    unique_category = df_train.category.unique().tolist()

    if corr:

        text_class.calc_corr(X_train_feat, df_train.category, unique_category, tf)

    if project:

        logging.info('************************* Project and plot 5 most frequent document categories in train set **********************************')
        
        if os.path.isfile('projected.png'):

            img = mpimg.imread('projected.png')

            imgplot = plt.imshow(img)

            plt.show()

        else:

            common = df_train.category.value_counts()[:5].index.tolist()

            labels_plot = [i for i in unique_category if i in common]

            text_class.project_features(X_train_feat, labels_plot, df_train.category)

    logging.info('************************* Run models **********************************')

    md = Model(X_train_feat, y_train, X_test_feat, y_test, balance)

    md.run_models()

    print("Execution time: {}".format(time.time() - start_time))





