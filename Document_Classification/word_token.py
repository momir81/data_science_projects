import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import logging
logging.basicConfig(level = logging.INFO)

class WordToken(object):
    '''
    Performs cleaning, preparation and tokenization of text in documents. 

    Arguments:
        train_set and test_set

    '''

    def __init__(self,df_train,df_test):

        self.df_train = df_train

        self.df_test = df_test
        

    def word_tokenize(self,text):
        ''''
        
        This method uses nltk library to translate data to tokens, convert to lower case, remove 
        punctation, non alphabetic characters, remove stopwords and lemmatize words.
        
        input: text document
        
        output: words for each document
        '''
        tokens = word_tokenize(text)

        # convert to lower case
        tokens = [w.lower() for w in tokens]

        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)

        stripped = [w.translate(table) for w in tokens]

        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]

        # filter out stop words
        stop_words = set(stopwords.words('english'))

        words = [w for w in words if not w in stop_words]
        
        #lemmatize is time consuming so that is comment out
        
        #lemmatizer = WordNetLemmatizer()
        
        #lemm_words = [lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in words]
        
        return words
    
    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)  

    def join_strings(self,row):

        '''
        Word_tokenize returns list of tokens such as:
        ['constantia',
        'flexibles',
        'aim',
        'double',
        'sale']. 
        So we need to convert those tokens to strings for extracting features
        using tfidf method in this shape 'constantia flexibles aim double sale'.

        '''
        return ' '.join(row)

    def run_tokenize(self):

        logging.info('Cleaning and tokenization of training data')

        self.df_train['text'] = self.df_train['text'].apply(lambda x:self.word_tokenize(x))

        self.df_train['joined'] = self.df_train.text.apply(self.join_strings)

        logging.info('Cleaning and tokenization of test data')

        self.df_test['text'] = self.df_test['text'].apply(lambda x:self.word_tokenize(x))
        
        self.df_test['joined'] = self.df_test.text.apply(self.join_strings)

        return self.df_train, self.df_test