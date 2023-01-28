from darts import TimeSeries
from darts.models import NBEATSModel

from forecaster.Test_Models.TestModel import TestModel


class NBeats(TestModel):
    def __init__(self, input_chunk_length=14, output_chunk_length=1, n_epochs=5, random_state=0):
        super().__init__(
            "N-Beats_with_inputChunkLength_{}_outputChunkLength_{}_nEpochs_{}_randomState_{}"
            .format(input_chunk_length, output_chunk_length, n_epochs, random_state))
        self.model = NBEATSModel(input_chunk_length=input_chunk_length,
                                 output_chunk_length=output_chunk_length,
                                 batch_size=14 * 4,
                                 n_epochs=200,
                                 force_reset=True,
                                 save_checkpoints=True,
                                 pl_trainer_kwargs={"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}
                                 )
        self.is_model_trained = False

    def train_model(self, train, column_name):
        return

    def get_prediction(self, train, test, column_name):
        if not self.is_model_trained:
            self.model.fit(TimeSeries.from_series(train[column_name]),
                           val_series=TimeSeries.from_series(test[column_name]), verbose=0)

        predictions = self.model.predict(n=len(test), series=TimeSeries.from_series(train[column_name])).pd_dataframe()

        self.is_model_trained = True
        return predictions[column_name]

    def run_model(self, train, test, column_name):
        if not self.is_model_trained:
            self.model.fit(TimeSeries.from_series(train[column_name]),
                           val_series=TimeSeries.from_series(test[column_name]), verbose=0)

        predictions = self.model.predict(n=len(test), series=TimeSeries.from_series(train[column_name])).pd_dataframe()

        return predictions[column_name]
