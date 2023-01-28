import pandas as pd
from darts import TimeSeries
from darts.models import LightGBMModel

from forecaster.Test_Models.TestModel import TestModel


class LightGBM(TestModel):

    def __init__(self, input_chunk_length=14, output_chunk_length=1, n_epochs=100, random_state=0):
        super().__init__(
            "LightGBM_with_inputChunkLength_{}_outputChunkLength_{}_nEpochs_{}_randomState_{}"
            .format(input_chunk_length, output_chunk_length, n_epochs, random_state))
        self.model = LightGBMModel(
            lags=1,
            random_state=24,
            likelihood="poisson",
            output_chunk_length=1,
            multi_models=True
        )
        self.is_model_trained = False
        self.n_steps = 14
        self.n_features = 1

    def train_model(self, train, column_name):
        train_copy = train.copy(deep=True)
        self.model.fit(TimeSeries.from_series(train_copy[column_name]), verbose=0)
        self.is_model_trained = True

    def get_prediction(self, train, test, column_name):
        train_copy = train.copy(deep=True)
        test_copy = test.copy(deep=True)

        last_n_records_in_train = train_copy.iloc[-len(test_copy):]
        new_test = pd.concat([last_n_records_in_train, test_copy], ignore_index=True)

        predictions = self.model.predict(n=len(test_copy), series=TimeSeries.from_series(new_test[column_name]), verbose=0).pd_dataframe()
        predictions.columns = ["{}_Predicted".format(column_name)]
        return predictions["{}_Predicted".format(column_name)]

    def run_model(self, train, test, column_name):
        if not self.is_model_trained:
            self.model = LightGBMModel(
                lags=1,
                random_state=24,
                likelihood="poisson",
                output_chunk_length=1,
                multi_models=True
            )
        self.model.fit(TimeSeries.from_series(train[column_name]), val_series=TimeSeries.from_series(test[column_name]),
                       verbose=0)

        predictions = self.model.predict(n=len(test), series=TimeSeries.from_series(train[column_name]),
                                         verbose=0).pd_dataframe()

        return predictions[column_name]
