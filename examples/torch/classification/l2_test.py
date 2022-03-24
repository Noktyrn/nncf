import torch
from nncf.torch.quantization.quantize_functions import QuantizeAsymmetric, QuantizeSymmetric, _quantize_autograd_to_range
from torch import nn
import numpy as np
from pot_quantizer import FakeQuantize
from torch.optim import SGD
from torch.nn import MSELoss
from nncf.torch.utils import sum_like

# reference impl
class ReferenceQuantizeSymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, scale, bits):
        print("Started applying reference quantize to {}".format(input_))
        print("Scale equals: {}".format(scale))
        level_high = scale.new_tensor(2 ** (bits - 1) - 1)
        level_low = scale.new_tensor(-(level_high + 1))
        s = level_high / scale
        print("Scale transformed equals: {}".format(s))

        output = input_ * s
        print("Scaled input: {}".format(output))
        output = output.clamp(min=level_low, max=level_high)
        print("Clamped scaled input: {}".format(output))
        output = output.round()
        print("Rounded clamped scaled input: {}".format(output))
        output = output / s
        print("Descaled input: {}".format(output))

        ctx.save_for_backward(input_, scale, output)
        ctx.level_high = level_high
        ctx.level_low = level_low

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, scale, output = ctx.saved_tensors
        level_high = ctx.level_high
        level_low = ctx.level_low

        alpha = float(level_low) / float(level_high)
        mask_hi = (input_ > scale).type(input_.dtype)
        mask_lo = (input_ < scale * alpha).type(input_.dtype)
        mask_in = 1 - mask_hi - mask_lo

        val_grad_out = mask_hi + alpha * mask_lo
        err = (output - input_) * scale.reciprocal()
        grad_scale = grad_output * (err * mask_in + val_grad_out)
        grad_scale = sum_like(grad_scale, scale)

        # calc gradient for input
        grad_input = grad_output * mask_in

        return grad_input, grad_scale, None

w = None
with open('test_x.npy', 'rb') as f:
    w = np.load(f)
x = None
with open('test_x.npy', 'rb') as f:
    x = np.load(f)
w = torch.Tensor(w)
print("W original: {}".format(w))
x = torch.Tensor(x)
y = torch.matmul(x, w)

min_val = torch.full_like(w, -1)
max_val = torch.full_like(w, 1)
levels = 255
bits = 8

class NNCFquant(torch.nn.Module):
    def __init__(self, w, min_val, max_val, levels):
        super(NNCFquant, self).__init__()
        self.weights = torch.nn.Parameter(w, requires_grad = True)
        self.min = min_val
        ranges = np.array(max_val, dtype=np.float32)
        self.scales = torch.nn.Parameter(torch.tensor(ranges).log(), requires_grad = True)
        self.levels = levels
    
    def forward(self, x):
        w = self.weights
        s = self.scales.exp()
        l = self.levels
        w_quant = ReferenceQuantizeSymmetric.apply(w, s, l)
        return torch.matmul(x, w_quant)

class POTquant(torch.nn.Module):
    def __init__(self, w, min_val, max_val, levels):
        super(POTquant, self).__init__()
        self.weights = torch.nn.Parameter(w, requires_grad = True)
        self.fq = FakeQuantize(min_val, max_val, levels)

    def forward(self, x):
        w_quant = self.fq(self.weights)
        return torch.matmul(x, w_quant)

model_nncf = NNCFquant(w, min_val, max_val, bits)
optimizer_nncf = SGD(model_nncf.parameters(), lr=1e-4)
loss_nncf = MSELoss()

"""
for _ in range(1):
    loss1 = loss_nncf(y, model_nncf(x))
    loss1.backward()
    optimizer_nncf.step()
"""
w_quant_nncf = ReferenceQuantizeSymmetric.apply(model_nncf.weights, 
                                                model_nncf.scales.exp(),
                                                model_nncf.levels)


with open('test_x.npy', 'rb') as f:
    w = np.load(f)
w = torch.Tensor(w)

model_pot = POTquant(w, min_val, max_val, levels)
optimizer_pot = SGD(model_pot.parameters(), lr=1e-4)
loss_pot = MSELoss()
"""
for _ in range(1):
    loss2 = loss_pot(y, model_pot(x))
    loss2.backward()
    optimizer_pot.step()
"""

w_quant_pot = model_pot.fq(model_pot.weights)
print("W pot: {}".format(w_quant_pot))

print("W nncf: {}".format(w_quant_nncf))
