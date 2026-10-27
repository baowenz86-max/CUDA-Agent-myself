import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        lstm_output = cuda_extension.lstm_forward(
            x,
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
            self.lstm.bias_ih_l0,
            self.lstm.bias_hh_l0,
            self.lstm.weight_ih_l1,
            self.lstm.weight_hh_l1,
            self.lstm.bias_ih_l1,
            self.lstm.bias_hh_l1,
            x.shape[0],
            x.shape[1],
            x.shape[2],
            self.lstm.hidden_size,
            self.lstm.num_layers
        )
        last_step = lstm_output[-1]
        fc_output = cuda_extension.linear_forward(
            last_step,
            self.fc.weight,
            self.fc.bias,
            self.fc.out_features
        )
        exp_output = cuda_extension.exp_forward(fc_output)
        return exp_output

    def debug_forward(self, x):
        lstm_output = cuda_extension.lstm_forward(
            x,
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
            self.lstm.bias_ih_l0,
            self.lstm.bias_hh_l0,
            self.lstm.weight_ih_l1,
            self.lstm.weight_hh_l1,
            self.lstm.bias_ih_l1,
            self.lstm.bias_hh_l1,
            x.shape[0],
            x.shape[1],
            x.shape[2],
            self.lstm.hidden_size,
            self.lstm.num_layers
        )
        yield ("step_001", (lstm_output, (torch.zeros_like(self.lstm.bias_ih_l0), torch.zeros_like(self.lstm.bias_ih_l0))))
        last_step = lstm_output[-1]
        fc_output = cuda_extension.linear_forward(
            last_step,
            self.fc.weight,
            self.fc.bias,
            self.fc.out_features
        )
        yield ("step_002", fc_output)
        exp_output = cuda_extension.exp_forward(fc_output)
        yield ("step_003", exp_output)
        yield ("step_004", exp_output)
        return exp_output
