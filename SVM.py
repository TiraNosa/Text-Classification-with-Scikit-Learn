# https://www.kaggle.com/lbronchal/sentiment-analysis-with-svm
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import utility

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer as stemmer
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

trainingData = "V1.4_Training_new.csv"
testData = "SubtaskB_EvaluationData_labeled.csv"

data_path = trainingData
data_path2 = testData
out_path = trainingData[:-4] + "_predictions.csv"
out_path2 = testData[:-4] + "_SVM_TfidfVectorizer_min_df=3_sublineartf_norm='l2'_test_size=0.2_random_state=1.csv"

data = pd.read_csv(trainingData)
data_clean = data.copy()
# data_clean = data_clean[data_clean['airline_sentiment_confidence'] > 0.65]

# We use the BeautifulSoup library to process html encoding present in some tweets because scrapping.
data_clean['text_clean'] = data_clean['sentence'].apply(lambda x: BeautifulSoup(x, "lxml").text)
data_clean = data_clean.loc[:, ['text_clean', 'label']]
pd.set_option('display.max_colwidth', -1)  # Setting this so we can see the full content of cells
# data_clean['text_clean'] = data_clean.text_clean.apply(normalizer)
# print(data_clean[['text_clean']].head())

train, test = train_test_split(data_clean, test_size=0.2, random_state=1)
X_train = train['text_clean'].values
X_test = test['text_clean'].values
y_train = train['label']
y_test = test['label']


def tokenize(text):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

en_stopwords = set(stopwords.words("english"))

vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    analyzer='word',
    tokenizer=tokenize,
    lowercase=True,
    ngram_range=(1, 1),
    stop_words=en_stopwords,
    norm='l2',
    min_df=3)

features = vectorizer.fit_transform(data['sentence']).toarray()
labels = data['label']
print(features.shape)
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

np.random.seed(1)

pipeline_svm = make_pipeline(vectorizer,
                             SVC(probability=True, kernel="linear", class_weight="balanced"))

model = GridSearchCV(pipeline_svm,
                     param_grid={'svc__C': [0.01, 0.1, 1]},
                     cv=kfolds,
                     scoring="roc_auc",
                     verbose=1,
                     n_jobs=1)
model.fit(X_train, y_train)
model.score(X_test, y_test)


def classify(sent_list):
    label_list = []
    for sent in sent_list:
        label_list.append(model.predict([sent[1]])[0])

    return label_list


if __name__ == '__main__':
    sent_list = utility.read_csv(testData)
    label_list = classify(sent_list)
    utility.write_csv(sent_list, label_list, out_path2)
