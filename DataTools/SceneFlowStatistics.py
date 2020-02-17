import argparse
import numpy as np
import os

from CommonPython.Filesystem.FileRecorder import read_string_list
from CommonPython.Filesystem.Filesystem import get_filename_parts

from IO import readPFM as read_disparity_file

def statistics_of_disparity_file(fn):
    # Test the file.
    if ( not os.path.isfile(fn) ):
        raise Exception("%s does not exist." % (fn))

    # Read the disparity.
    disp, scale = read_disparity_file(fn)

    # Statistics.
    dispMax  = disp.max()
    dispMin  = disp.min()
    dispMean = disp.mean()

    return dispMin, dispMax, dispMean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the statistics of the Scene Flow dataset.")

    parser.add_argument("inputdir", type=str, \
        help="The directory contains the file lists.")
    
    parser.add_argument("--left-disparity-list", type=str, default="InputDisparityTest.txt")

    args = parser.parse_args()

    # Read the file list.
    fnList = read_string_list("%s/%s" % ( args.inputdir, args.left_disparity_list ))

    N = len( fnList )

    print("%d files in the list." % (N))

    statArray = []

    # Process each file.
    for fn in fnList:
        print(fn)
        dispMin, dispMax, dispMean = statistics_of_disparity_file(fn)

        statArray.append( [dispMin, dispMax, dispMean] )

    # Save the statistics.
    parts = get_filename_parts(args.left_disparity_list)
    print(parts)

    statArray = np.array(statArray, dtype=np.float32)
    np.savetxt("%s/%s_Stat.txt" % (args.inputdir, parts[1]), statArray)

