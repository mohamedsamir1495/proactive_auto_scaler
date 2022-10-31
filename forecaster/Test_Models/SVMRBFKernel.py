import numpy as np
from sklearn.svm import SVR

from forecaster.Test_Models.TestModel import TestModel


class SVMRBFKernel(TestModel):
    def __init__(self):
        super().__init__("SVM RBF Kernel")
        self.model = None

    def train_model(self, train, column_name):
        svr_norm = SVR(kernel="rbf", C=1.9, epsilon=0.027, gamma=1)
        X = np.arange(0, len(train), 1).reshape(-1, 1)
        X %= 24

        svr_fit = svr_norm.fit(X, train[column_name])
        self.model = svr_fit

    def get_prediction(self, train, test, column_name):
        test_copy = test.copy(deep=True)
        test_copy["{}_Predicted".format(column_name)] = self.model.predict((np.arange(len(train), len(train) + len(test), 1).reshape(-1, 1)) % 24)

        return test_copy["{}_Predicted".format(column_name)]

    def run_model(self, train, test, column_name):
        svr_norm = SVR(kernel="rbf", C=1.9, epsilon=0.027, gamma=1)
        X = np.arange(0, len(train), 1).reshape(-1, 1)
        X %= 24

        svr_fit = svr_norm.fit(X, train[column_name])
        test['prediction'] = svr_fit.predict((np.arange(len(train), len(train) + len(test), 1).reshape(-1, 1)) % 24)

        return test["prediction"]
