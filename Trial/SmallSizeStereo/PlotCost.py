#!/env/python

import matplotlib.pyplot as plt
import numpy as np
import os

def save_layer(fn, layer):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(layer)
    fig.savefig(fn)
    plt.close(fig)

if __name__ == "__main__":
    # Load the cost file.
    cost = np.load("cost.npy")

    print("cost.shape = {}".format( cost.shape ))

    outDir = "CostLayers"

    # Test the output directory.
    if ( not os.path.isdir(outDir) ):
        os.makedirs( outDir )

    # Plot the cost layers.
    n = cost.shape[1]

    for i in range(n):
        print(i)
        
        # Get the layer.
        layer = cost[0, i, :, :]

        # Output filename.
        fn = "%s/%04d.png" % (outDir, i)

        # Save the layer.
        save_layer(fn, layer)
