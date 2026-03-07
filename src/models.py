import time
import numpy as np
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.interval_based import QUANTClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.deep_learning import LITETimeClassifier
from sklearn.metrics import accuracy_score

def get_classifiers(seed):
    return {
        "1NN-DTW": KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw"),
        "Rocket": RocketClassifier(random_state=seed),
        "QUANT": QUANTClassifier(random_state=seed),
        "LITE": LITETimeClassifier(random_state=seed)
    }

def train_and_evaluate_classifier(clf_name, clf, X_train, y_train, X_test, y_test):
    print(f'Trainning {clf_name}...')
    start = time.time()
    clf.fit(X_train, y_train)
    duration = np.round(time.time() - start, 2)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, duration