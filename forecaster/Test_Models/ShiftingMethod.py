from forecaster.Test_Models.TestModel import TestModel


class ShiftingMethod(TestModel):
    def __init__(self, shift_window):
        super().__init__("shifting_method_with_{}_shift_window".format(shift_window))
        self.shift_window = shift_window

    def run_model(self, train, test, column_name):
        predicted_column_name = "{}_Predicted".format(column_name)

        test[predicted_column_name] = test[column_name].shift(self.shift_window)
        test[predicted_column_name][0:self.shift_window] = train[column_name][-self.shift_window:]

        return test[predicted_column_name]
