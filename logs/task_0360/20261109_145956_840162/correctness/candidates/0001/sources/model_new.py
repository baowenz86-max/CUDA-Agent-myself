import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        # 保持与参考模型相同的 state_dict 键
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, bias=bias)

    def forward(self, x: torch.Tensor):
        # 与 nn.RNN 一致，输入 x shape (seq, batch, input)
        seq_len, batch, _ = x.shape
        hidden_size = self.hidden_size
        num_layers = self.num_layers
        device = x.device
        dtype = x.dtype

        # 初始化 h_0 为零向量
        h_0 = torch.zeros(num_layers, batch, hidden_size, device=device, dtype=dtype)

        # 存储每一层每一步的输出
        output = torch.empty(seq_len, batch, hidden_size, device=device, dtype=dtype)
        h_n = torch.empty(num_layers, batch, hidden_size, device=device, dtype=dtype)

        # 将输入和权重移到与 x 相同的设备（验证时输入 x 为 cuda，模型参数会由验证器移到 cuda）
        # 注意：模型参数存储在 self.rnn 中，我们直接使用 self.rnn 的参数
        weight_ih_l0 = self.rnn.weight_ih_l0
        weight_hh_l0 = self.rnn.weight_hh_l0
        bias_ih_l0 = self.rnn.bias_ih_l0 if self.bias else torch.zeros(hidden_size, device=device, dtype=dtype)
        bias_hh_l0 = self.rnn.bias_hh_l0 if self.bias else torch.zeros(hidden_size, device=device, dtype=dtype)

        # 第一层输入为原始输入 x，逐时间步计算
        layer_input = x
        for layer in range(num_layers):
            if layer == 0:
                weight_ih = weight_ih_l0
                weight_hh = weight_hh_l0
                bias_ih = bias_ih_l0
                bias_hh = bias_hh_l0
            else:
                weight_ih = self.rnn.weight_ih_l1
                weight_hh = self.rnn.weight_hh_l1
                bias_ih = self.rnn.bias_ih_l1 if self.bias else torch.zeros(hidden_size, device=device, dtype=dtype)
                bias_hh = self.rnn.bias_hh_l1 if self.bias else torch.zeros(hidden_size, device=device, dtype=dtype)

            # 当前层的隐状态，初始为零
            h = h_0[layer]  # shape [batch, hidden]
            layer_output = torch.empty(seq_len, batch, hidden_size, device=device, dtype=dtype)

            for t in range(seq_len):
                input_t = layer_input[t]  # [batch, input_size]
                # 调用自定义算子完成单步 RNN 计算
                h = cuda_extension.rnn_cell_forward(
                    input_t, h, weight_ih, weight_hh, bias_ih, bias_hh
                )
                layer_output[t] = h

            # 保存该层输出作为下一层输入
            layer_input = layer_output
            output = layer_output if layer == num_layers - 1 else output
            h_n[layer] = h

        return output, h_n

    def debug_forward(self, *inputs):
        # 按 checkpoint 顺序 yield，最后 yield 最终返回
        # 注意：debug_forward 也必须使用自定义算子，不能调用 torch 算子
        # 这里复用 forward 逻辑，但通过 yield 返回中间结果
        x = inputs[0]
        seq_len, batch, _ = x.shape
        hidden_size = self.hidden_size
        num_layers = self.num_layers
        device = x.device
        dtype = x.dtype

        h_0 = torch.zeros(num_layers, batch, hidden_size, device=device, dtype=dtype)
        output = torch.empty(seq_len, batch, hidden_size, device=device, dtype=dtype)
        h_n = torch.empty(num_layers, batch, hidden_size, device=device, dtype=dtype)

        weight_ih_l0 = self.rnn.weight_ih_l0
        weight_hh_l0 = self.rnn.weight_hh_l0
        bias_ih_l0 = self.rnn.bias_ih_l0 if self.bias else torch.zeros(hidden_size, device=device, dtype=dtype)
        bias_hh_l0 = self.rnn.bias_hh_l0 if self.bias else torch.zeros(hidden_size, device=device, dtype=dtype)

        layer_input = x
        for layer in range(num_layers):
            if layer == 0:
                weight_ih = weight_ih_l0
                weight_hh = weight_hh_l0
                bias_ih = bias_ih_l0
                bias_hh = bias_hh_l0
            else:
                weight_ih = self.rnn.weight_ih_l1
                weight_hh = self.rnn.weight_hh_l1
                bias_ih = self.rnn.bias_ih_l1 if self.bias else torch.zeros(hidden_size, device=device, dtype=dtype)
                bias_hh = self.rnn.bias_hh_l1 if self.bias else torch.zeros(hidden_size, device=device, dtype=dtype)

            h = h_0[layer]
            layer_output = torch.empty(seq_len, batch, hidden_size, device=device, dtype=dtype)
            for t in range(seq_len):
                input_t = layer_input[t]
                h = cuda_extension.rnn_cell_forward(
                    input_t, h, weight_ih, weight_hh, bias_ih, bias_hh
                )
                layer_output[t] = h
            layer_input = layer_output
            output = layer_output if layer == num_layers - 1 else output
            h_n[layer] = h

        result = (output, h_n)
        yield 'step_001', result
        return
