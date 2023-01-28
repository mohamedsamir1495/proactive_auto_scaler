import warnings

from darts import TimeSeries
from darts.models import NHiTSModel
from numpy import array

from forecaster.Test_Models.TestModel import TestModel

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence.iloc[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


class NHits(TestModel):

    def __init__(self, input_chunk_length=14, output_chunk_length=1, n_epochs=100, random_state=0):
        super().__init__(
            "N-Hits_with_inputChunkLength_{}_outputChunkLength_{}_nEpochs_{}_randomState_{}"
            .format(input_chunk_length, output_chunk_length, n_epochs, random_state))
        self.model = NHiTSModel(input_chunk_length=input_chunk_length,
                                output_chunk_length=output_chunk_length,
                                batch_size=14 * 4,
                                n_epochs=200,
                                force_reset=True,
                                save_checkpoints=True,
                                pl_trainer_kwargs={"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}
                                )
        self.is_model_trained = False
        self.n_steps = 14
        self.n_features = 1

    def train_model(self, train, column_name):
        # self.model.fit(train, verbose=0)
        return

    def get_prediction(self, train, test, column_name):
        if not self.is_model_trained:
            self.model.fit(TimeSeries.from_series(train[column_name]),
                           val_series=TimeSeries.from_series(test[column_name]), verbose=0)

        self.is_model_trained = True
        predictions = self.model.predict(n=len(test), series=TimeSeries.from_series(train[column_name])).pd_dataframe()

        return predictions[column_name]

    def run_model(self, train, test, column_name):
        if not self.is_model_trained:
            self.model = NHiTSModel(input_chunk_length=14,
                                    output_chunk_length=1,
                                    dropout=0.1,
                                    batch_size=14 * 4,
                                    n_epochs=200,
                                    force_reset=True,
                                    save_checkpoints=True,
                                    pl_trainer_kwargs={"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}
                                    )
            self.model.fit(TimeSeries.from_series(train[column_name]),
                           val_series=TimeSeries.from_series(test[column_name]), verbose=0)

        predictions = self.model.predict(n=len(test), series=TimeSeries.from_series(train[column_name]),
                                         verbose=0).pd_dataframe()

        return predictions[column_name]
