from functools import wraps
import torch


# Since Batchnorm requires at least single precision floats,
#   to test float16 and lower we upcast into the golden model and downcast when returning the results
def upcast_half_or_quarter_precision_to_float32(model_fn):
    @wraps(model_fn)
    def wrapper_fn(*args, dtype, **kwargs):
        if torch.finfo(dtype).bits < torch.finfo(torch.float32).bits:
            print("upcasting precision to float32")
            upcasted_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    upcasted_args.append(arg.to(torch.float32))
                else:
                    upcasted_args.append(arg)
            upcasted_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    upcasted_kwargs[k] = v.to(torch.float32)
                else:
                    upcasted_kwargs[k] = v
            upcasted_output = model_fn(dtype=dtype, *upcasted_args, **upcasted_kwargs)
            if isinstance(upcasted_output, tuple):
                downcasted_output = []
                for output in upcasted_output:
                    if isinstance(output, torch.Tensor):
                        downcasted_output.append(output.to(dtype))
                    else:
                        downcasted_output.append(output)
                return tuple(downcasted_output)
            else:
                return upcasted_output.to(dtype)
        else:
            return model_fn(dtype=dtype, *args, **kwargs)

    return wrapper_fn


def golden_model_forward_eval(
    ifmap, eps, running_mean, running_var, weight, bias, *, dtype
) -> torch.Tensor:
    n, ci, ih, iw = ifmap.shape
    bn = torch.nn.BatchNorm2d(ci, eps, dtype=dtype)
    bn.weight = torch.nn.Parameter(weight)
    bn.bias = torch.nn.Parameter(bias)
    bn.running_mean = running_mean
    bn.running_var = running_var
    bn.eval()
    return bn(ifmap)


def golden_model_forward_training(
    ifmap, eps, running_mean, running_var, weight, bias, momentum, *, dtype
) -> torch.Tensor:
    n, ci, ih, iw = ifmap.shape
    bn = torch.nn.BatchNorm2d(ci, eps, momentum=momentum, dtype=dtype)
    bn.weight = torch.nn.Parameter(weight)
    bn.bias = torch.nn.Parameter(bias)
    bn.running_mean = running_mean
    bn.running_var = running_var
    ofmap = bn(ifmap)
    return ofmap, bn.running_mean, bn.running_var


@upcast_half_or_quarter_precision_to_float32
def golden_model_backward(
    ifmap, grad_ofmap, weight, bias, running_mean, running_var, eps, *, dtype
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    n, ci, ih, iw = ifmap.shape
    bn = torch.nn.BatchNorm2d(ci, eps=eps, dtype=dtype)
    bn.weight = torch.nn.Parameter(weight)
    bn.bias = torch.nn.Parameter(bias)
    bn.running_mean = running_mean.clone()
    bn.running_var = running_var.clone()
    bn.eval()
    ofmap = bn(ifmap)
    ofmap.retain_grad()
    ifmap.retain_grad()
    bn.weight.retain_grad()
    ofmap.flatten().dot(grad_ofmap.flatten()).backward()
    return ifmap.grad, bn.weight.grad, bn.bias.grad


def golden_model_backward_training(
    ifmap, grad_ofmap, weight, bias, eps, *, dtype
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    n, ci, ih, iw = ifmap.shape
    bn = torch.nn.BatchNorm2d(ci, eps=eps, dtype=dtype)
    bn.weight = torch.nn.Parameter(weight)
    bn.bias = torch.nn.Parameter(bias)
    ofmap = bn(ifmap)
    ofmap.retain_grad()
    ifmap.retain_grad()
    bn.weight.retain_grad()
    ofmap.flatten().dot(grad_ofmap.flatten()).backward()
    return ifmap.grad, bn.weight.grad, bn.bias.grad


# Implementation of backprop for batchnorm training mode without autograd
def my_golden_model_backward_training(
    ifmap, grad_ofmap, weight, bias, current_mean, current_var, eps, *, dtype
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    n, ci, ih, iw = ifmap.shape
    num_points = n * ih * iw
    invstd = torch.rsqrt(current_var + eps)
    sum = grad_ofmap.sum(dim=(0, 2, 3))
    dotp = torch.zeros(ci)
    for c in range(ci):
        dotp[c] = torch.sum(
            grad_ofmap[:, c, :, :] * ((ifmap[:, c, :, :] - current_mean[c]))
        )
    k = dotp * invstd * invstd / num_points
    grad_mean = sum / num_points
    dx = torch.zeros(ifmap.shape)
    for c in range(ci):
        dx[:, c, :, :] = (ifmap[:, c, :, :] - current_mean[c]) * k[c]
    grad_ifmap = torch.zeros(ifmap.shape)
    for c in range(ci):
        grad_ifmap[:, c, :, :] = (
            (grad_ofmap[:, c, :, :] - grad_mean[c] - dx[:, c, :, :])
            * invstd[c]
            * weight[c]
        )
    grad_weight = dotp * invstd
    grad_bias = sum
    return grad_ifmap, grad_weight, grad_bias
