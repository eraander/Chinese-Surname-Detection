#-----------------------------------------------------------
# Place name classification module
# By: Qingwen Ye @ 12/19/2018
#-----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import OneHotEncoder

KEYS = ['prev_direction', 'post_direction', 'prev_tag', 'post_tag', 'post_adm',
    'prev_visit', 'adm', 'direction', 'geo', 'char']


class rule_template(object):
    '''
    A base class for rule based classification
    '''
    def __init__(self):
        return None

    def pred_all(self, X_test):
        return [self.pred(x) for x in X_test]


class rule_based_clf(rule_template):
    '''
    original rule based classfication, used as the baseline
    '''
    def __init__(self):
        return None

    def pred(self, features):
        '''
        rule-based prediction
        Args:
            features: input features
        '''
        result = 0

        if features['adm'] == 1:
            result = 1

        if features['geo'] == 1:
            result = 1

        if features['post_direction'] == 1 or features['prev_visit'] == 1:
            result = 1

        return result


class rule_based_refine(rule_template):
    '''
    refined rule-based classification classifier, combine rules extracted from
    the decision tree and human-designed rules
    '''
    def __init__(self):
        return None

    def pred(self, features):
        result = 0

        if features['adm'] == 0:
            if features['direction'] == 0:
                if features['geo'] == 0:
                    if features['char'] == 0: return 0
                    if features['char'] == 1: return 1
                if features['geo'] == 1:
                    if features['post_tag'] == 'IDIOMS': return 0
                    if features['prev_tag'] == 'ADV': return 0
                    return 1
            if features['direction'] == 1: return 1
        if features['adm'] == 1: return 1

        return result


rule_names = ["Original rule based", "Refined rule based"]

rule_classifiers = [
    rule_based_clf(),
    rule_based_refine(),
]

ml_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes"]

ml_classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),]


def eval_res(pred, actual):
    '''
    Evaluate the prediction and actual label, return the metrics including
    accuracy, true negative, false positive, false negative, and true positive
    Args:
        pred: list of predicted values
        actual: list of actual labels
    '''
    accuracy = sklearn.metrics.accuracy_score(actual, pred)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(actual, pred).ravel()
    return round(accuracy, 3), tn, fp, fn, tp


def plot_all_res(all_res):
    '''
    plot the comparison amoing different models
    Args:
        all_res: list of metrics in the seq of [accuracy, tn, fp, fn, tp]
    '''
    acc = [element[0] for element in all_res]
    tn = [element[1] for element in all_res]
    fp = [element[2] for element in all_res]
    fn = [element[3] for element in all_res]
    tp = [element[4] for element in all_res]

    objects = rule_names + ml_names

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(objects))
    ax0 = axs[0]
    ax0.bar(x, acc, width=0.25, color='grey')
    ax0.set_ylabel('Accuracy')
    ax0.set_xlabel('Models')
    ax0.set_title('Accuracy comparison')
    ax0.set_xticks(x)
    ax0.set_xlabel('Models')
    ax0.set_xticklabels(['rule\n based', 'rule based\n refine', "nearest\n neighbors",
        "linear\n SVM", "RBF\n SVM", "decision\n tree", "random\n forest",
        "neural\n net", "adaBoost", "naive\n Bayes"], fontsize=8)

    ax1 = axs[1]
    ax1.bar(x-0.2, tn, width=0.15, label='tn')
    ax1.bar(x-0.05, fp, width=0.15, label='fp')
    ax1.bar(x+0.1, fn, width=0.15, label='fn')
    ax1.bar(x+0.25, tp, width=0.15, label='tp')
    ax1.set_ylabel('#Samples')

    ax1.set_xticks(x-0.2, objects)
    ax1.set_title('Other metrics')
    ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels(['rule\n based', 'rule based\n refine', "nearest\n neighbors",
        "linear\n SVM", "RBF\n SVM", "decision\n tree", "random\n forest",
        "neural\n net", "adaBoost", "naive\n Bayes"], fontsize=8)

    plt.show()


def eval_classifier(training_set, test_set):
    '''
    evaluate the performance of different classifiers
    Args:
        training_set: list of training instances in the format of [feature, label]
        test_set: list of test instances in the format of [feature, label]

    '''
    X_test = [element[0] for element in test_set]
    Y_test = [element[1] for element in test_set]
    all_res = []

    for name, clf in zip(rule_names, rule_classifiers):
        Y_pred = clf.pred_all(X_test)
        accuracy, tn, fp, fn, tp = eval_res(Y_pred, Y_test)
        print('{} model has test accuracy: {}, tn: {}, fp: {}, fn: {}, tp: {}'
            .format(name, accuracy, tn, fp, fn, tp))

        all_res.append([accuracy, tn, fp, fn, tp])

    X = [[element[0][k] for k in KEYS] for element in training_set]
    Y = [element[1] for element in training_set]

    X_test = [[element[0][k] for k in KEYS] for element in test_set]
    Y_test = [element[1] for element in test_set]

    # One hot encoding is required for machine learning model
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc.fit(X, Y)
    X = enc.transform(X)
    X_test = enc.transform(X_test)

    for name, clf in zip(ml_names, ml_classifiers):
        clf.fit(X, Y)
        
        Y_pred = clf.predict(X_test)
        accuracy, tn, fp, fn, tp = eval_res(Y_pred, Y_test)
        print('{} model has test accuracy: {}, tn: {}, fp: {}, fn: {}, tp: {}'
            .format(name, accuracy, tn, fp, fn, tp))
        all_res.append([accuracy, tn, fp, fn, tp])

    return all_res
