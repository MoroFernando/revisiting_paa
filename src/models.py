import time
import numpy as np
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.interval_based import QUANTClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.deep_learning import LITETimeClassifier
from sklearn.metrics import accuracy_score

def get_classifier_instance(name, seed):
    clfs = {
        "1NN-DTW": lambda: KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw"),
        "Rocket": lambda: RocketClassifier(random_state=seed),
        "QUANT": lambda: QUANTClassifier(random_state=seed),
        "LITE": lambda: LITETimeClassifier(random_state=seed)
    }
    return clfs[name]() if name in clfs else None

def train_and_evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    start = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    duration = np.round(time.time() - start, 2)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, duration