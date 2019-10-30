from __future__ import print_function

from multiprocessing import Process
import os
import numpy as np
import torch

def func(name, n=4):
    np.random.seed( int(os.getpid()) )
    torch.manual_seed( int(os.getpid()) )

    for i in range(n):
        print("%s: %d, np.random.randn(1) = %+f, torch.randn(1) = %+f. " % ( name, i, np.random.randn(1), torch.randn(1) ))

if __name__ == "__main__":
    processes = []
    processes.append( Process(target=func, args=("p1", 4)) )
    processes.append( Process(target=func, args=("p2", 4)) )

    print("Start all the processes.")

    for p in processes:
        p.start()
    
    for p in processes:
        p.join()

    print("All processes joined.")