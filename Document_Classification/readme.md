# Project: Classify text documents

## Approached methodology:

1. Clean and preprocess raw text documents
2. Convert preprocessed documents to word embedding
3. Optionally calculate most correlated unigrams and bigrams in each document
4. Optionally project 5 most frequent documents to analyze their relationship
5. Optionally apply feature selection
6. Optionally apply upsampling document categories
7. Run model

1. In first step nltk library is used for cleaning and preprocessing. Lemmatization step is commented
out since it is time consuming and does not affect much accuracy statistics but from the other side it
does provide more meaningful insights into the documents.

2. In this step tfidf is used for calculating word weights and creating embedding matrix. In this step also
it was applied variety of other approaches but tfidf was chosen due to simplicity and efficiency. Some
of the experimented approaches included gensim word2vec, pretrained glove word embedding model
combined with keras embedding laayer and convolutional neural network.

3. For word correlation chi squared method is used.

4. For projection tsne is chosen.

5. For feature selection chi squared is used.

6. For upsampling minority classes RandomOverSample is used since it only worked with highly
sparse data. Unfortunately this step does not bring much improvement, but it is recommended in case
of highly skewed categories. It may give improvement if train and test sets would be design differently.

7. LinearSVC model is chosen due to simplicity and efficiency. Also it was experimented with other
models such as RidgeClassifier, RandomForest, MultinomialBN, LogisticRegression, SGDClassifier,
deep convolutional neural network combined with lstm layer, as well as ULMFiT fast ai model.

## Results:

Document similarity for 5 most common classes:
![Word documents similarity](https://github.com/momir81/data_science_projects/Document_Classification)

As we see document contents are very similar which affects their inference.

Precision score: 0.680741471910584
Recall score: 0.6803844998498048
F1 score: 0.6693633743953245
```
              precision    recall  f1-score   support

           0       0.80      0.87      0.84       173
           1       0.71      0.64      0.67       154
           2       0.00      0.00      0.00         1
           3       0.54      0.37      0.44        81
           4       0.63      0.66      0.65        94
           5       0.65      0.62      0.64        24
           6       1.00      0.11      0.20         9
           7       0.00      0.00      0.00         2
           8       0.69      0.71      0.70       205
           9       1.00      0.54      0.70        13
          10       0.83      0.19      0.32        77
          11       0.65      0.57      0.61        49
          12       0.54      0.46      0.49        48
          13       0.67      0.72      0.70       243
          14       0.29      0.17      0.21        12
          15       0.75      0.43      0.55         7
          16       0.70      0.65      0.67       134
          17       0.88      0.73      0.80        59
          18       1.00      0.38      0.56        13
          19       0.83      0.77      0.80        39
          20       0.00      0.00      0.00         5
          21       0.83      0.38      0.52        50
          22       0.69      0.82      0.75       286
          23       0.65      0.55      0.60        31
          24       0.67      0.79      0.72       303
          25       0.68      0.81      0.74       176
          26       0.62      0.68      0.65        19
          27       0.64      0.74      0.69       409
          28       0.74      0.62      0.68        32
          29       1.00      0.50      0.67         4
          30       0.61      0.50      0.55        34
          31       0.70      0.77      0.73       221
          32       0.43      0.21      0.29        14
          33       0.62      0.47      0.53       134
          34       0.63      0.61      0.62       163
          35       0.00      0.00      0.00        11

    accuracy                           0.68      3329
   macro avg       0.63      0.50      0.54      3329
weighted avg       0.68      0.68      0.67      3329
```
Due to imbalanced document classes and high document content similarity we see associated accuracy metrics.
For higher support (document label presence) we see higher F1 scores.

## Usability

Arguments used from command line:
–corr corr for running step 3
--project project for running step 4
--balance balance for running step 6
--corr None for running without including steps 3, 4, 5

*Code is packed in a docker image.*

1. From the same directory where Dockerfile is situated, run from command line sudo docker build -t
docs_classification_app .
2. Use for running plain code without optional arguments: sudo docker run docs_classification_app –corr None
3 Use any of the combinations from sudo docker run docs_classification_app –corr corr –project project –balance balance
4 Optional copyng saved models and files from docker container to local directory:
5. Find stopped containers: sudo docker ps -a
6. Copy: sudo docker cp container_id:/app <absolute_path_to_local_directory>
