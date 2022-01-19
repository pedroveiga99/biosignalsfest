import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit  # cross-validation with random splits
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score



features = pd.read_csv('features_tudo.csv')

dic_state = {
    'rest_pressure': 1,
    'medium_pressure': 2,
    'high_pressure': 3
}

# Convert the pressure states to numbers
features['state'] = features['state'].apply(lambda x: dic_state[x])

# Filter the features with high correlation
state_corr = features.corr()['state'].abs()
relevant_features = state_corr[state_corr > 0.3]

feat_names = []
for col in relevant_features.index:
    feat_names.append(col)

features = features[feat_names]  # Only data from the relevant features

# Classificadores
clf_randforest = RandomForestClassifier(max_depth=7)
clf_knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
clf_naive = GaussianNB()
clf_neural = MLPClassifier(activation='logistic')
clf_svm = svm.SVC()

train, test = train_test_split(features, test_size=0.2, shuffle=True)  # train -> 60% | test -> 40%

for clf in (clf_randforest, clf_naive, clf_knn, clf_neural, clf_svm):
    clf.fit(train.drop('state', axis=1), train['state'])

    predicted = clf.predict(test.drop('state', axis=1))
    disp = metrics.ConfusionMatrixDisplay.from_predictions(test['state'], predicted)
    disp.figure_.suptitle(clf)

    plt.show()
