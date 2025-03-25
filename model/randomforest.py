import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from numpy import *
import random
num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest:
    def __init__(self, model_name, embeddings):
        self.model_name = model_name
        self.embeddings = embeddings
        self.model = RandomForestClassifier(n_estimators=1000, random_state=0)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def print_results(self, predictions, y_test):
        #print(classification_report(y_test, predictions))
        accuracy = accuracy_score(y_test, predictions)
        #print(f'Accuracy score: {accuracy}')
        return accuracy
    
    def data_transform(self) -> None:
        ...

