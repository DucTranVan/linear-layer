import linear
import numpy as np

def make_sample_data(size=100, n_features=3, n_targets=2):
    """
    The function is to generating data for experiments
    Args:
        size: int, the number of samples
        n_features: int, the number of features of each sample input
        n_tagets: int, the dimention of the output vector associated with a sample
    Return:
        X, Y, weights : list of numpy array where X is independent variable 
        and Y is dependent variable of X, weights the coefs of the underline truth function
    """
    generator = np.random.RandomState(42)
    X = generator.randn(size,n_features)
    weights = generator.rand(n_features, n_targets)
    Y = np.dot(X, weights)
    return X, Y, weights

# make an sample data
x_train, y_train, ground_truth = make_sample_data()

# set hyperparameters for training
n_iterates = 1000
learning_rate = 0.01
# initilize weights
weights = np.random.rand(len(x_train[0]), len(y_train[0]))
# training loop
for i in range(n_iterates):
    for m in range(len(x_train)):
        # forward 
        sample = np.reshape(x_train[m],(1, x_train[m].size))
        y_pre = linear.linear_transform_numpy_array(weights,sample)
        # gradient descent updating
        delta = y_pre - y_train[m]
        update_matrix = np.dot(sample.T, delta)*learning_rate
        weights = weights - update_matrix
    # calculate mean square loss :
    y_pre = linear.linear_transform_numpy_array(weights,x_train)
    delta = (y_pre - y_train)
    loss = np.sum(delta**2, axis=0)
    loss = loss / len(x_train)
    print(i)
    print("------------")
    print(loss)
    print( weights)
    print(ground_truth)


