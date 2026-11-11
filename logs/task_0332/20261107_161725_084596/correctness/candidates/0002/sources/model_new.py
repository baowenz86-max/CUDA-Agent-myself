import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, pooling_length):
        super().__init__()
        # Must preserve exact state_dict keys: adaptive_avg_pool has no parameters, lstm and fc have their own.
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(pooling_length)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * pooling_length, output_size)

    def forward(self, x):
        # Step 1: adaptive average pooling, output shape (batch, input_size, pooling_length)
        x = cuda_extension.adaptive_avg_pool1d_forward(x, self.pooling_length)

        # Step 2: transpose dims 1 and 2 to obtain (batch, pooling_length, input_size)
        x = cuda_extension.transpose_2d_forward(x, 1, 2)

        # Step 3: LSTM, batch_first=True, only the output sequence is used; hidden states are discarded exactly like x, _ = self.lstm(x)
        x, _ = cuda_extension.lstm_forward(x, self.lstm)

        # Step 4: flatten all dimensions after the batch dimension
        x = cuda_extension.flatten_forward(x, 1)

        # Step 5: fully connected layer
        x = cuda_extension.linear_forward(x, self.fc.weight, self.fc.bias)

        return x

    def debug_forward(self, *inputs):
        x = inputs[0]

        x = cuda_extension.adaptive_avg_pool1d_forward(x, self.pooling_length)
        yield "step_001", x

        x = cuda_extension.transpose_2d_forward(x, 1, 2)
        yield "step_002", x

        x, _ = cuda_extension.lstm_forward(x, self.lstm)
        yield "step_003", (x, _)

        x = cuda_extension.flatten_forward(x, 1)
        yield "step_004", x

        x = cuda_extension.linear_forward(x, self.fc.weight, self.fc.bias)
        yield "step_005", x

        yield "step_006", x