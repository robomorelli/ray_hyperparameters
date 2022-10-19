from torch.utils.data import Dataset
import torch

class Dataset_seq(Dataset):
    # TODO: implementation of recontruction and prediction (given the time steps recontruct all the
    #   features or predict the features dropped from the df and compare the prediction with the actual value
    #   for the anomaly detection task.

    # TODO: implement also the forecasting (the idx of target is shifted ahead of many steps of the forecasting window
    def __init__(self, df, target = None, sequence_length=4, out_window = 4,
                 prediction = False, forecast = False):

        self.prediction = prediction
        self.forecast = forecast
        #TODO raise error if prediction == true but target is not defined
        if (self.prediction) and (not self.forecast):
            self.df_data = df.drop(target, axis=1)
            self.targets = df[target]
        elif self.forecast:
            self.df_data = df
            self.targets = df[target]
        else: # In case of recontruction
            self.df_data = df  # In case of recontruction
            self.targets = df  # In case of recontruction

        self.sequence_length = sequence_length
        self.out_window = out_window

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        if self.forecast:
            if (idx + self.sequence_length + self.out_window) > len(self.df_data):
                indexes = list(range(len(self.df_data) - self.sequence_length, len(self.df_data)))
                indexes_out = list(range(len(self.df_data), len(self.df_data) + self.out_window))
            else:
                indexes = list(range(idx, idx + self.sequence_length))
                indexes_out = list(range(idx + self.sequence_length, idx + self.sequence_length + self.out_window))
        else:
            if (idx + self.sequence_length) > len(self.df_data):
                indexes = list(range(len(self.df_data) - self.sequence_length, len(self.df_data)))
            else:
                indexes = list(range(idx, idx + self.sequence_length))

            if (idx + self.out_window) > len(self.df_data):
                indexes_out = list(range(len(self.df_data) - self.out_window, len(self.df_data)))
            else:
                indexes_out = list(range(idx, idx + self.out_window))

        data = self.df_data.iloc[indexes, :].values
        target = self.targets.iloc[indexes_out].values
        return torch.tensor(data).float(), torch.tensor(target).float()  #torch.from_numpy(x).float()