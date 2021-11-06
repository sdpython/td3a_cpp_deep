"""
Unit tests.
"""
import unittest
import torch
from td3a_cpp_deep.fcts.piecewise_linear import (
    PiecewiseLinearFunction,
    PiecewiseLinearFunctionC,
    PiecewiseLinearFunctionCBetter)


class TestFctsPiecewiseLinear(unittest.TestCase):

    def test_equal_forward(self):
        alpha_pos = torch.tensor([1], dtype=torch.float32)
        alpha_neg = torch.tensor([0.5], dtype=torch.float32)
        x = torch.tensor([-2, 1], dtype=torch.float32)
        res1 = PiecewiseLinearFunction.apply(x, alpha_neg, alpha_pos)
        res2 = PiecewiseLinearFunctionC.apply(x, alpha_neg, alpha_pos)
        res3 = PiecewiseLinearFunctionCBetter.apply(x, alpha_neg, alpha_pos)
        for a, b, c in zip(res1, res2, res3):
            na = a.cpu().detach().numpy().tolist()
            nb = b.cpu().detach().numpy().tolist()
            nc = c.cpu().detach().numpy().tolist()
            self.assertEqual(na, nb)
            self.assertEqual(na, nc)

    def piecewise_linear(self, cls, device, verbose=False, max_iter=400):

        x = torch.randn(100, 1, device=device, dtype=torch.float32)
        y = x * 0.2 + (x > 0).to(torch.float32) * x * 1.5

        alpha_pos = torch.tensor([1], dtype=torch.float32).to(device)
        alpha_neg = torch.tensor([0.5], dtype=torch.float32).to(device)
        alpha_pos.requires_grad_()
        alpha_neg.requires_grad_()

        losses = []
        learning_rate = 1e-4
        fct = cls.apply

        for t in range(max_iter):

            y_pred = fct(x, alpha_neg, alpha_pos)
            loss = (y_pred - y).pow(2).sum()
            loss.backward()
            losses.append(loss)

            with torch.no_grad():
                if verbose:
                    print(alpha_neg.grad, alpha_pos.grad)
                alpha_pos -= learning_rate * alpha_pos.grad
                alpha_neg -= learning_rate * alpha_neg.grad

                # Manually zero the gradients after updating weights
                alpha_pos.grad.zero_()
                alpha_neg.grad.zero_()

        if max_iter > 300:
            self.assertTrue(losses[-1] < 1)
            self.assertTrue(abs(alpha_neg - 0.2) < 0.2)
            self.assertTrue(abs(alpha_pos - 1.7) < 0.2)

    def test_piecewise_linear_cpu(self):
        self.piecewise_linear(PiecewiseLinearFunction, torch.device('cpu'))

    @unittest.skipIf(not torch.cuda.is_available(), reason="no GPU")
    def test_piecewise_linear_gpu(self):
        self.piecewise_linear(PiecewiseLinearFunction, torch.device("cuda:0"))

    def test_piecewise_linear_c_cpu(self):
        self.piecewise_linear(PiecewiseLinearFunctionC, torch.device('cpu'))

    @unittest.skipIf(not torch.cuda.is_available(), reason="no GPU")
    def test_piecewise_linear_c_gpu(self):
        self.piecewise_linear(PiecewiseLinearFunctionC, torch.device("cuda:0"))

    def test_piecewise_linear_c_cpu_better(self):
        self.piecewise_linear(
            PiecewiseLinearFunction, torch.device('cpu'),
            verbose=True, max_iter=3)
        self.piecewise_linear(
            PiecewiseLinearFunctionCBetter, torch.device('cpu'),
            verbose=True, max_iter=3)
        self.piecewise_linear(
            PiecewiseLinearFunctionCBetter, torch.device('cpu'))

    @unittest.skipIf(not torch.cuda.is_available(), reason="no GPU")
    def test_piecewise_linear_c_gpu_better(self):
        self.piecewise_linear(
            PiecewiseLinearFunctionCBetter, torch.device("cuda:0"))


if __name__ == '__main__':
    unittest.main()
