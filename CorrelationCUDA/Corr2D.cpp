#include <torch/extension.h>

#include <vector>

// CUDA test interfaces.
torch::Tensor from_BCHW_2_BHWC_padded_cuda( torch::Tensor input, int padding );

// CUDA module interfaces.
std::vector<torch::Tensor> corr_2d_forward_cuda( torch::Tensor input );
std::vector<torch::Tensor> corr_2d_backward_cuda( torch::Tensor grad, torch::Tensor s );

// C++ interfaces.

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor test_from_BCHW_2_BHWC_padded(torch::Tensor input, int padding)
{
    CHECK_INPUT(input);

    return from_BCHW_2_BHWC_padded_cuda(input, padding);
}

std::vector<torch::Tensor> corr_2d_forward( torch::Tensor input )
{
    CHECK_INPUT(input);

    return corr_2d_forward_cuda(input);
}

std::vector<torch::Tensor> corr_2d_backward( torch::Tensor grad, torch::Tensor s )
{
    CHECK_INPUT(grad);
    CHECK_INPUT(s);

    return corr_2d_backward_cuda(grad, s);
}

PYBIND11_MODULE( TORCH_EXTENSION_NAME, m )
{
    m.def("test_from_BCHW_2_BHWC_padded", &test_from_BCHW_2_BHWC_padded, "TF: test_from_BCHW_2_BHWC_padded. ");
    m.def("forward", &corr_2d_forward, "Corr2D forward, CUDA version. ");
    m.def("backward", &corr_2d_backward, "Corr2D backward, CUDA version. ");
}

