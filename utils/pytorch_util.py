import torch
import torch.nn as nn
import numpy as np


def check_cuda_availability(cuda):
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("Error: You set cuda=True but no cuda device found.")


def try_tensor_cuda(tensor, cuda):
    if cuda and torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor.cpu()