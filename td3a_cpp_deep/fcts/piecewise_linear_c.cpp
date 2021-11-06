#include <torch/extension.h>
#include <iostream>

std::vector<at::Tensor> piecewise_linear_forward(
        torch::Tensor input,
        torch::Tensor alpha_neg,
        torch::Tensor alpha_pos) {

    // python code
    // sign = (input >= 0).to(torch.float32)
    // weight = (sign * alpha_pos + (- sign + 1) * alpha_neg)
    // output = input * weight

    auto sign = (input >= 0).to(torch::kFloat32);
    auto weight = (sign * alpha_pos) + (- sign + 1) * alpha_neg;
    return {input * weight, input, sign, weight};
}


std::vector<at::Tensor> piecewise_linear_forward_better(
        torch::Tensor input,
        torch::Tensor alpha_neg,
        torch::Tensor alpha_pos) {

    auto sign = (input >= 0).to(torch::kFloat32);
    auto weight = sign * (alpha_pos - alpha_neg);
    weight += alpha_neg;
    return {input * weight, input, sign, weight};
}


std::vector<torch::Tensor> piecewise_linear_backward(
        torch::Tensor grad_output,
        torch::Tensor input,
        torch::Tensor sign,
        torch::Tensor weight) {

    // python code
    // grad_input = weight
    // grad_alpha_neg = (input * grad_output * (- sign + 1)).sum(dim=0, keepdim=True)
    // grad_alpha_pos = (input * grad_output * sign).sum(dim=0, keepdim=True)
    // return grad_input, grad_alpha_neg, grad_alpha_pos

    auto grad_alpha_neg = at::sum((input * grad_output * (- sign + 1)), {0}, true);
    auto grad_alpha_pos = at::sum((input * grad_output * sign), {0}, true);
    return {weight, grad_alpha_neg, grad_alpha_pos};
}


std::vector<torch::Tensor> piecewise_linear_backward_better(
        torch::Tensor grad_output,
        torch::Tensor input,
        torch::Tensor sign,
        torch::Tensor weight) {

    auto ig = input * grad_output;
    auto igs = ig * sign;
    auto grad_alpha_pos = at::sum(igs, {0}, true);
    igs *= -1;
    igs += ig;
    auto grad_alpha_neg = at::sum(igs, {0}, true);
    return {weight, grad_alpha_neg, grad_alpha_pos};
}


PYBIND11_MODULE(piecewise_linear_c, m) {
	m.doc() =
    #if defined(__APPLE__)
    "C++ Implementation of piecise linear function."
    #else
    R"pbdoc(C++ Implementation of piecise linear function.)pbdoc"
    #endif
    ;

    m.def("piecewise_linear_forward", &piecewise_linear_forward, "PiecewiseLinearC forward");
    m.def("piecewise_linear_backward", &piecewise_linear_backward, "PiecewiseLinearC backward");

    m.def("piecewise_linear_forward_better", &piecewise_linear_forward_better,
          "PiecewiseLinearC improved forward");
    m.def("piecewise_linear_backward_better", &piecewise_linear_backward_better,
          "PiecewiseLinearC improved backward");
}
