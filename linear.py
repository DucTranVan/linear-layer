def linear_transform_py_list(weight_matrix, data):
    """
    The linear transform function implement using python data structure.
    Input is multiple vectors, represented by a matrix which each row is an sample.
    Output is a matrix which each row represents transform result for each sample,
    each colum represents an particular transform node responding for the whole data.
    Args:
        weight_matrix : The 2 dimentional list represents weights for transform data
        data : the 2 dimentional list which each row is an sample to transform
    Return:
        out_layer: The 2 dimentional list which the first dimention is for each sample in input data,
        the length of the second dimention respond to the number of ouput nodes.
    """
    m = len(data)
    n = len(data[0])
    k = len(weight_matrix)
    assert n == len(weight_matrix[0])
    out_layer = [[0 for j in range(k)] for i in range(m)]
    for i in range(m):
        for j in range(k):
            for h in range(n):
                out_layer[i][j] += data[i][h]*weight_matrix[j][h]
    return out_layer

def linear_transform_numpy_array(weight_matrix, data):
    """
    The linear transform function implement using numpy.
    Args:
        weight_matrix: 2d numpy array, each colum is an weight vector using to calculate each 
        element in output layer.
        data: 2d numpy array, each row is an sample input to transform
    Return:
        output_layer: 2d array which each colum responds to each node in output layer 
    """
    import numpy as np 
    (m,n) = data.shape
    (h ,k) = weight_matrix.shape
    assert n == h
    out_layer = data.dot(weight_matrix)
    return out_layer

