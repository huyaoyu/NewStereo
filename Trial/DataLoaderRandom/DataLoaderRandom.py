from __future__ import print_function

import numpy as np
import os

import torch
import torch.utils.data as data

class RandomDataFolder(data.Dataset):
    def __init__(self, n=3):
        super(RandomDataFolder, self).__init__()

        self.data = [ i for i in range(n) ]

    def __len__(self):
        return len( self.data )

    def __getitem__(self, idx):

        a = torch.FloatTensor( (self.data[idx], ) )

        print("a[0] = %f, np.random.rand(1) = %f, torch.randn(1) = %+f, os.getpid() = %d. " % (a[0], np.random.rand(1), torch.randn(1), os.getpid()))

        return a

    def show(self):
        for d in self.data:
            print(d)

if __name__ == "__main__":
    print("Test the random functions with dataloader.")

    # Create the dataset.
    dataset = RandomDataFolder(8)

    print("The original data: ")
    dataset.show()
    print("")

    # Create the dataloader.
    dataloader = data.DataLoader( dataset, \
        batch_size=2, shuffle=False, num_workers=2, drop_last=False )
    
    # import ipdb; ipdb.set_trace()

    # Test.
    print("The actual loaded data.")
    for batchIdx, (data) in enumerate( dataloader ):
        for i in range(data.size()[0]):
            print("batchIdx = %d, data = %f. " % ( batchIdx, data[i, 0] ))
    