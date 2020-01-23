import numpy as np
import torch

def test_1():
    # Create two tensors.
    t0 = torch.ones((6,)).double()
    t1 = t0.clone()

    # Manually modify the values.
    t0[4] = -1.0
    t0[5] = -1.0

    t1[2] = -1.0
    t1[3] = -1.0

    # Turn on the gradient computation flag.
    t0.requires_grad = True
    t1.requires_grad = True

    # Create a ReLU object.
    MReLU = torch.nn.ReLU()

    # Show details of computation.
    print("t0 = \n{}".format(t0))

    r0 = MReLU(t0)
    print("r0 = \n{}".format(r0))

    r0.backward( torch.ones((6,)) )
    print("t0.grad = \n{}".format(t0.grad))

    # t1.
    print("t1 = \n{}".format(t1))

    r1 = MReLU(t1)
    print("r1 = \n{}".format(r1))

    r1.backward( torch.ones((6,)) )
    print("t1.grad = \n{}".format(t1.grad))

def test_2():
    # Create two tensors.
    t0 = torch.ones((6,)).double()
    t1 = t0.clone()

    # Manually modify the values.
    t0[4] = -1.0
    t0[5] = -1.0

    t1[2] = -1.0
    t1[3] = -1.0

    # Turn on the gradient computation flag.
    t0.requires_grad = True
    t1.requires_grad = True

    # Create a ReLU object.
    MReLU = torch.nn.ReLU()

    # Show details of computation.
    print("t0 = \n{}".format(t0))

    r0 = MReLU(t0)
    print("r0 = \n{}".format(r0))

    # t1.
    print("t1 = \n{}".format(t1))

    r1 = MReLU(t1)
    print("r1 = \n{}".format(r1))

    # Compute the gradients.
    r0.backward( torch.ones((6,)) )
    print("t0.grad = \n{}".format(t0.grad))

    r1.backward( torch.ones((6,)) )
    print("t1.grad = \n{}".format(t1.grad))

if __name__ == "__main__":
    test_2()