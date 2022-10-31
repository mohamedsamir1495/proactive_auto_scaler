class TestModel:
    def __init__(self, name):
        self.name = name

    def train_model(self, train, column_name):  # train model only
        pass

    def get_prediction(self, train, test, column_name):  # test model only
        pass

    def run_model(self, train, test, column_name):  # train and test model in one go
        pass

    def __str__(self):
        return self.name
