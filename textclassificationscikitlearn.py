import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import utility


class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


def readFile(file):
    df = pd.read_csv(file)
    df.head()

    # add a column encoding the label as an integer
    # because categorical variables are often better represented by integers than strings
    col = ['label', 'sentence']
    df = df[col]
    df = df[pd.notnull(df['sentence'])]
    df.columns = ['label', 'sentence']
    df['category_id'] = df['label']
    df['label_str'] = df['label'].replace({1: 'suggestion',0: 'non-suggestion'})
    df.head()
    return df


trainData = 'V1.4_Training_new.csv'
testData = 'SubtaskB_EvaluationData_labeled.csv'
out_path = testData[:-4] + "_RandomForestClassifier.csv"

df = readFile(trainData)
category_id_df = df[['label_str', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'label_str']].values)
# imbalanced classes
# imbalanced classes
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')
features = tfidf.fit_transform(df.sentence).toarray()
labels = df.category_id
print(features.shape)

# find the terms that are the most correlated with each of the labels
# from sklearn.feature_selection import chi2
import numpy as np
#
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns

# Model Evaluation
model = RandomForestClassifierWithCoef()
X_train, X_val, y_train, y_val, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                               test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

model.fit(features, labels)

N = 50
for Label, category_id in sorted(category_to_id.items()):
    model.fit(features, labels == category_id)
    indices = np.argsort(model.coef_)
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1][:N]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2][:N]
    print("# '{}':".format(Label))
    print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
    print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))

from sklearn import metrics

# evaluation data prediction
df = readFile(testData)
category_id_df = df[['label_str', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'label_str']].values)
features = tfidf.transform(df.sentence).toarray()
print(features.shape)
y_pred = model.predict(features).astype('int')
y_val = df.label

fig = plt.figure(figsize=(4, 3))
df.groupby('label_str').sentence.count().plot.bar(ylim=0)
plt.show()

print(df.groupby('label_str').sentence.count())

if __name__ == '__main__':
    sent_list = utility.read_csv(testData)
    label_list = y_pred
    utility.write_csv(sent_list, label_list, out_path)

print(metrics.classification_report(y_val, y_pred, target_names=df['label_str'].astype('str').unique()))
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_val, y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.label_str.values, yticklabels=category_id_df.label_str.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()