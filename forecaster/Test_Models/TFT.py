from darts import TimeSeries
from darts.models import TFTModel
import pandas as pd

from forecaster.Test_Models.TestModel import TestModel

class TFT(TestModel):
    def __init__(self, input_chunk_length=14, output_chunk_length=1, n_epochs=5, random_state=0):
        super().__init__(
            "TFT_with_inputChunkLength_{}_outputChunkLength_{}_nEpochs_{}"
            .format(input_chunk_length, output_chunk_length, n_epochs, random_state))
        self.model = TFTModel(input_chunk_length=1,
                              output_chunk_length=1,
                              dropout=0.1,
                              lstm_layers=4,
                              batch_size=14 * 4,
                              n_epochs=200,
                              force_reset=True,
                              save_checkpoints=True,
                              add_relative_index=True
                              ,
                              pl_trainer_kwargs={"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}
                              )
        self.is_model_trained = False

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
            self.model = TFTModel(input_chunk_length=14,
                                  output_chunk_length=1,
                                  dropout=0.1,
                                  lstm_layers=4,
                                  batch_size=14 * 4,
                                  n_epochs=200,
                                  force_reset=True,
                                  save_checkpoints=True,
                                  add_relative_index=True,
                                  pl_trainer_kwargs={"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}
                                  )
            self.model.fit(TimeSeries.from_series(train[column_name]),
                           val_series=TimeSeries.from_series(test[column_name]), verbose=0)
        predictions = self.model.predict(n=len(test), series=TimeSeries.from_series(train[column_name])).pd_dataframe()

        return predictions[column_name]