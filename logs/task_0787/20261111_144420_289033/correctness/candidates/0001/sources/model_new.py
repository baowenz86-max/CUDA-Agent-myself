import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, padding_size, padding_value):
        super().__init__()
        # 参考模型没有可学习参数，state_dict 为空
        # 保留 padding_size 和 padding_value 以便 forward 使用
        self.padding_size = padding_size
        self.padding_value = padding_value

    def forward(self, x):
        # 第一步：沿通道维 softmax
        x = torch.ops.cuda_extension.custom_softmax_dim1(x)
        # 第二步：常数填充 2D
        # padding_size 参考模型 nn.ConstantPad2d 接受 (pad_left, pad_right, pad_top, pad_bottom) 或 (pad_w, pad_h)
        # 任务中 padding_size=(1,2)，根据输出 shape [16,64,512,515] 推断实际为左右各1、上下无填充？
        # 但任务契约明确输出 [16,64,512,515]，宽度增加3？需要仔细确定。
        # 标准 ConstantPad2d((1,2)) 表示左右各1，上下各2，输出 [16,64,516,514]。
        # 而契约输出 [16,64,512,515]，宽 515 = 512+3，高不变，说明实际 padding 是 (1,2) 被解释为左侧1、右侧2，上下0？
        # 但 ConstantPad2d 的输入若为 (1,2) 作为 2-tuple 表示 (left, right)，4-tuple 才是 (left, right, top, bottom)。
        # 这里 padding_size 是 (1,2)，2-tuple，所以 pad_left=1, pad_right=2, pad_top=0, pad_bottom=0，输出 [16,64,512,515]。
        # 但参考模型 __init__ 传入 padding_size=(1,2) 且 ConstantPad2d 接受 2-tuple 或 4-tuple，2-tuple 时表示左右，因此输出宽度 512+1+2=515，高度不变，与契约一致。
        # 因此 padding 参数应展开为 (1,2,0,0)
        pad_left, pad_right = self.padding_size
        pad_top, pad_bottom = 0, 0
        # 调用自定义填充算子，参数顺序：x, padding_left, padding_right, padding_top, padding_bottom, value
        x = torch.ops.cuda_extension.custom_constant_pad2d(x, pad_left, pad_right, pad_top, pad_bottom, self.padding_value)
        return x

    def debug_forward(self, *inputs):
        # 按照参考检查点顺序 yield (id, value)
        x = inputs[0]
        # step_001: softmax
        x = torch.ops.cuda_extension.custom_softmax_dim1(x)
        yield "step_001", x
        # step_002: pad
        pad_left, pad_right = self.padding_size
        x = torch.ops.cuda_extension.custom_constant_pad2d(x, pad_left, pad_right, 0, 0, self.padding_value)
        yield "step_002", x
        # step_003: final return
        yield "step_003", x
