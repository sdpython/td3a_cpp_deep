"""
Piecewise linear functions.
"""
import torch
from torch.autograd import Function


class PiecewiseLinearFunction(Function):

    @staticmethod
    def forward(ctx, input, alpha_neg, alpha_pos):
        sign = (input >= 0).to(torch.float32)
        weight = (sign * alpha_pos + (- sign + 1) * alpha_neg)
        ctx.save_for_backward(input, alpha_neg, alpha_pos, sign, weight)
        output = input * weight
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha_neg, alpha_pos, sign, weight = ctx.saved_tensors
        grad_input = weight
        grad_alpha_neg = (
            input * grad_output * (- sign + 1)).sum(dim=0, keepdim=True)
        grad_alpha_pos = (
            input * grad_output * sign).sum(dim=0, keepdim=True)
        return grad_input, grad_alpha_neg, grad_alpha_pos
