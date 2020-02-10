from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from word_token import WordToken
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging
logging.basicConfig(level = logging.INFO)


class TextClassify(object):
    '''
    Perform text analysis and word correlations for each document. Finaly, it classifies all documents and 
    reports accuracy statistics. 

    Arguments:
        train data filepath: string 
        test data filepath: string
    '''

    def __init__(self, df_train, df_test):

        self.df_train = df_train

        self.df_test = df_test

    
    def calculate_tfidf(self):

        word_token = WordToken(self.df_train, self.df_test)

        self.df_train, self.df_test = word_token.run_tokenize()

        docs_train = self.df_train['joined']

        docs_test = self.df_test['joined']
    
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1',ngram_range=(1,2), stop_words='english')
    
        # Fit TfIdf model
        
        # Transform a document into TfIdf coordinates

        logging.info('Tfidf transform train data')
        
        X_train = tfidf.fit_transform(docs_train).toarray()

        logging.info('Tfidf transform test data')
        
        X_test = tfidf.transform(docs_test).toarray()
        
        return X_train, X_test, tfidf
    
    def encode_labels(self, labels):

        labelencoder = LabelEncoder()

        labels = labelencoder.fit_transform(labels)

        return labels

    def get_features(self, X_train, X_test, y_train):
    
        ch2 = SelectKBest(chi2, k=10000)
        
        X_train = ch2.fit_transform(X_train, y_train)
        
        X_test = ch2.transform(X_test)
        
        return X_train, X_test

    def calc_corr(self, features, labels, unique_category, tf):

        logging.info('******* Calculate most correlated unigrams and bigrams for train set *****************')

        N = 3

        for category in (unique_category):

            features_chi2 = chi2(features,  labels == category)

            indices = np.argsort(features_chi2[0])

            feature_names = np.array(tf.get_feature_names())[indices]

            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]

            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

            print("# '{}':".format(category))

            print(" . Most correlated unigrams:\n   .{}".format('\n        .'.join(unigrams[-N:])))

            print(" . Most correlated bigrams:\n   .{}".format('\n        .'.join(bigrams[-N:])))

    def project_features(self, features, labels_plot, labels):
    
        SAMPLE_SIZE = int(len(features) * 0.3)

        np.random.seed(0)

        plt.figure(figsize=(20,10))

        indices = np.random.choice(range(len(features)), size = SAMPLE_SIZE, replace=False)

        projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])

        colors = ['pink', 'green', 'blue', 'yellow', 'grey']

        i = 0

        for category in labels_plot:

            points = projected_features[(labels[indices] == category).values]
            
            plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[i], label=category)

            i += 1

        plt.title("tf-idf features vector for each article, projected on 2 dimensions.", fontdict=dict(fontsize=15))

        plt.legend()

        plt.show()

        plt.savefig('projected.png')

    
