from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
import numpy as np
import imblearn
import pickle
import os
import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level = logging.INFO)

class Model(object):

    '''
    This class perform modelling of text documents with and without upsampling and calculates 
    accuracy statistics.
    
    Arguments:
        required: X_train, y_train, X_test, y_test
        optional: balance

    '''

    def __init__(self, X_train, y_train, X_test, y_test, balance):

        self.X_train = X_train

        self.y_train = y_train

        self.X_test = X_test

        self.y_test = y_test

        self.balance_class = balance


    def run_models(self):

        '''
        Run models and calculates accuracy, precision, recall and F1 scores. After run it saves models for later use.
        '''

        if self.balance_class:

            self.X_train, self.y_train = self.balance(self.X_train, self.y_train)

            print("Balanced data shape: {}".format(self.X_train.shape))

        models = [LinearSVC(penalty='l2', dual=False,tol=1e-3)]

        for model in models:

            model_name = model.__class__.__name__

            print('Model: {}'.format(model_name))

            if self.balance_class:

                filename = model_name + '_resampled' +'.sav'

            else:

                filename = model_name + '.sav'

            if os.path.isfile(filename):

                model = pickle.load(open(filename, 'rb'))
            
            else:

                model.fit(self.X_train, self.y_train)

                pickle.dump(model, open(filename, 'wb'))

            predictions = model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test,predictions)

            print("Accuracy score: {}".format(accuracy))

            print("Precision score: {}".format(precision_score(self.y_test, predictions, average = 'weighted')))

            print("Recall score: {}".format(recall_score(self.y_test, predictions, average = 'weighted')))

            print("F1 score: {}".format(f1_score(self.y_test, predictions, average = 'weighted')))
            
            print(classification_report(self.y_test, predictions))
            
            #print(confusion_matrix(y_test, predictions))

            np.savetxt('confusion_mat.csv',confusion_matrix(self.y_test, predictions), delimiter=';')


    def balance(self, X, y):

        '''
        Run over-sampling method for balancing document categories.
        '''
        logging.info('************************* Balance text categories **********************************')

        sm = imblearn.over_sampling.RandomOverSampler(random_state=42)
        
        X_resampled, y_resampled = sm.fit_sample(X, y)
        
        return X_resampled, y_resampled

    
