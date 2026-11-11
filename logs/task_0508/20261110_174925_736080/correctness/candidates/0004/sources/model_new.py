import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # 参考 Model 仅有一个 BCEWithLogitsLoss，不包含任何可学习参数。
        # 为了保持 state_dict 为空，这里不注册任何 Parameter 或 buffer。

    def forward(self, y_pred, y_true):
        stacked_predictions = cuda_extension.stack_forward(y_pred)
        stacked_true_labels = cuda_extension.stack_forward(y_true)
        return cuda_extension.bce_with_logits_loss_forward(stacked_predictions, stacked_true_labels)

    def debug_forward(self, y_pred, y_true):
        stacked_predictions = cuda_extension.stack_forward(y_pred)
        yield 'step_001', stacked_predictions
        stacked_true_labels = cuda_extension.stack_forward(y_true)
        yield 'step_002', stacked_true_labels
        loss = cuda_extension.bce_with_logits_loss_forward(stacked_predictions, stacked_true_labels)
        yield 'step_003', loss
