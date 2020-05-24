import argparse
import linear
import numpy as np 



parser = argparse.ArgumentParser()
parser.add_argument('--mode',type=int, default=0,
                    help='type of implementation to test, 0 for python list implement, other for numpy array implement')
args = parser.parse_args()
data = [[1,2,3],[2,1,1]]
weight = [[1, 1, 1],[1,0,0]]
if args.mode == 0:
    out_layer = linear.linear_transform_py_list(weight,data)
    print(out_layer)
else:
    np_data = np.array(data)
    np_weight = np.array(weight).T
    out_layer = linear.linear_transform_numpy_array(np_weight,np_data)
    print(out_layer)
