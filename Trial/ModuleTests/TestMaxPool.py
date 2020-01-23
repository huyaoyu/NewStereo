import numpy as np
import torch

def test_1():
    # Create two tensors.
    t0 = torch.arange(6).double().view((1, 1, 6))
    t1 = t0.clone().flip([2])

    # Turn on the gradient computation flag.
    t0.requires_grad = True
    t1.requires_grad = True

    # Create a MaxPool object.
    MMP = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

    # Show details of computation.
    print("t0 = \n{}".format(t0))

    r0 = MMP(t0)
    print("r0 = \n{}".format(r0))

    r0.backward( torch.ones((1, 1, 3)) )
    print("t0.grad = \n{}".format(t0.grad))

    # t1.
    print("t1 = \n{}".format(t1))

    r1 = MMP(t1)
    print("r1 = \n{}".format(r1))

    r1.backward( torch.ones((1, 1, 3)) )
    print("t1.grad = \n{}".format(t1.grad))

def test_2():
    # Create two tensors.
    t0 = torch.arange(6).double().view((1, 1, 6))
    t1 = t0.clone().flip([2])

    # Turn on the gradient computation flag.
    t0.requires_grad = True
    t1.requires_grad = True

    # Create a MaxPool object.
    MMP = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

    # Show details of computation.
    print("t0 = \n{}".format(t0))

    r0 = MMP(t0)
    print("r0 = \n{}".format(r0))

    print("t1 = \n{}".format(t1))

    r1 = MMP(t1)
    print("r1 = \n{}".format(r1))

    r0.backward( torch.ones((1, 1, 3)) )
    print("t0.grad = \n{}".format(t0.grad))

    r1.backward( torch.ones((1, 1, 3)) )
    print("t1.grad = \n{}".format(t1.grad))
if __name__ == "__main__":
    test_2()