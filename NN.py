#from keras.datasets import mnist
import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle as pk


train_X = idx2numpy.convert_from_file('train-images.idx3-ubyte')
train_y = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
test_X = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
test_y = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')


train_X_f = train_X.reshape(train_X.shape[0],-1).T
train_X_f = train_X_f/255

train_y_f = train_y.reshape((1,60000))
train_y_a = np.zeros((10,60000),dtype=int)

test_x_f = test_X.reshape(test_X.shape[0],-1).T
test_x_f = test_x_f/255
test_y_f = test_y.reshape((1,10000))

ran_num = np.random.randint(60000,size=(10000))
trainX_acc = train_X_f[:,ran_num]
trainY_acc = train_y_f[:,ran_num]

def random_mini_batches(X,Y,batch_size,seed):
    """

    :param X:
    :param Y:
    :param batch_size:
    :return:
    """
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    perm = np.random.permutation(m)
    random_X = X[:,perm]
    random_Y = Y[:,perm]

    total_batches = m//batch_size
    for i in range(0,total_batches):
        batch_X = random_X[:,i*batch_size:(i+1)*batch_size]
        batch_Y = random_Y[:,i*batch_size:(i+1)*batch_size]
        batch = (batch_X,batch_Y)
        mini_batches.append(batch)

    if m%batch_size !=0:
        temp = (m//batch_size)*batch_size
        batch_X = random_X[:,temp:]
        batch_Y = random_Y[:,temp:]
        batch = (batch_X,batch_Y)
        mini_batches.append(batch)

    return mini_batches





def initialize_params(layer_dims):
    """

    :param layer_dims:
    :return:
    """
    np.random.seed(2)
    parameters = {}
    for l in range(1,len(layer_dims)):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def relu(Z):
    A = np.maximum(Z,0)
    return A,Z


def softmax(Z):
    #print('Z',Z)
    z_exp = np.exp(Z)
    z_sum = np.sum(z_exp,axis=0,keepdims=True)
    A = np.divide(z_exp,z_sum)
    return A,Z


def linear_forward(A_prev,W,b):
    """

    :param A_prev:
    :param W:
    :param b:
    :return:
    """

    Z = np.dot(W,A_prev) + b
    #linear_cache = (A_prev,W,b)
    linear_cache = (A_prev, W)
    return Z,linear_cache


def linear_activation_forward(A_prev,W,b,activation):
    """

    :param A_prev:
    :param W:
    :param b:
    :param activation:
    :return:
    """

    if activation == 'relu':
        Z,linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    elif activation == 'softmax':
        #print('W',W)
        #print('A_prev',A_prev)
        Z,linear_cache = linear_forward(A_prev,W,b)
        #print('Z',Z)
        #print('b',b)
        A, activation_cache = softmax(Z)

    cache = (linear_cache,activation_cache)
    return A,cache


def forward_model(X,parameters):
    """

    :param X:
    :param parameters:
    :return:
    """
    caches = []
    A = X
    L = len(parameters)//2

    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],activation='relu')
        caches.append(cache)

    AL,cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],activation='softmax')
    caches.append(cache)

    return AL,caches


def compute_cost(AL,Y):
    """

    :param AL:
    :param Y:
    :return:
    """
    m = Y.shape[1]
    temp = -np.sum(Y*np.log(AL),axis=0,keepdims=True)
    cost = np.sum(temp,axis=1)/m
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ,linear_cache):
    """

    :param dZ:
    :param linear_cache:
    :return:
    """
    #A_prev, W,b = linear_cache
    A_prev,W = linear_cache
    m = A_prev.shape[1]
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)
    return dA_prev,dW,db


def relu_backward(dA,activation_cache):
    """

    :param dA:
    :param activation_cache:
    :return:
    """
    Z = activation_cache
    gz = (Z>=0)*1
    dZ = dA*gz

    return dZ


def softmax_backwards(Y,AL):
    """

    :param dA:
    :param activation_cache:
    :return:
    """
    dZ = AL-Y

    return dZ


def linear_activation_backward(Y,AL,dA,cache,activation):
    """

    :param dA:
    :param cache:
    :param activation:
    :return:
    """
    linear_cache,activation_cache = cache
    if activation=='relu':
        dZ = relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)

    elif activation=='softmax':
        dZ = softmax_backwards(Y,AL)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)

    return dA_prev,dW,db


def backward_model(AL,Y,caches):
    """

    :param AL:
    :param Y:
    :param caches:
    :return:
    """
    grads = {}
    L = len(caches)



    dAL = -np.divide(Y,AL)
    #print('dAL',dAL)


    current_cache = caches[L-1]
    dA_prev,dW,db = linear_activation_backward(Y,AL,dAL,current_cache,activation='softmax')
    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    #('dA_prev',dA_prev)
    #print('dW',dW)
    #print('db',db)

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(Y,AL,grads['dA'+str(l+1)], current_cache, activation='relu')
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l+1)] = dW
        grads["db" + str(l+1)] = db

    return grads


def update_parameters(params,grads,learning_rate):
    parameters = params.copy()
    L = len(parameters)//2

    for l in range(L):
        parameters["W" + str(l + 1)] = params["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = params["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters





layers_dims = [784,300,100,30,10]

def L_layer_model(X,Y,layer_dims,epochs,learning_rate,num_iterations,print_cost):
    """

    :param X:
    :param Y:
    :param layer_dims:
    :param learning_rate:
    :param num_iterations:
    :param print_cost:
    :return:
    """

    #np.random.seed(1)
    train_y_a = np.zeros((10, 60000), dtype=int)
    for i in range(0, 60000):
        train_y_a[Y[0][i]][i] = 1
    m = X.shape[1]
    costs = []
    batchwise_accuracy = []
    parameters = initialize_params(layer_dims)
    seed = 5
    mini_batch_size = 128

    for ep in range(epochs):
        seed = seed+1
        mini_batches = random_mini_batches(X,train_y_a,mini_batch_size,seed)
        total_cost = 0

        for batch in mini_batches:
            (batch_X,batch_Y) = batch

            AL,caches = forward_model(batch_X,parameters)

            total_cost += compute_cost(AL,batch_Y)

            grads = backward_model(AL,batch_Y,caches)

            parameters = update_parameters(parameters,grads,learning_rate)

            temp = AL.shape[1]
            AL_max = np.max(AL,axis=0,keepdims=True)
            y_hat = np.zeros((1, temp))
            batch_Y_reduced = np.zeros((1,temp))
            for i in range(temp):
                for j in range(AL.shape[0]):
                    if AL[j][i] == AL_max[0][i]:
                        y_hat[0][i] = j
                        break
            for i in range(temp):
                for j in range(batch_Y.shape[0]):
                    if batch_Y[j][i] == 1:
                        batch_Y_reduced[0][i] = j
                        break
            crrt_prdt = (y_hat == batch_Y_reduced) * 1
            sum = np.sum(crrt_prdt)

            batch_accuracy = sum / temp
            batchwise_accuracy.append(batch_accuracy)


        avg_cost = total_cost/m



        print("Cost after iteration {}: {}".format(ep, np.squeeze(avg_cost)))
        #if ep % 100 == 0 or ep == num_iterations:
        costs.append(avg_cost)


    return parameters,costs,batchwise_accuracy


def predictions(X,Y,parameters):
    m = X.shape[1]
    AL,caches = forward_model(X,parameters)
    AL_max = np.max(AL,axis=0,keepdims=True)
    y_hat = np.zeros((1,m))

    for i in range(m):
        for j in range(AL.shape[0]):
            if AL[j][i]==AL_max[0][i]:
                y_hat[0][i] = j
                break

    crrt_prdt = (y_hat==Y)*1
    sum = np.sum(crrt_prdt)

    accuracy = sum/m
    return accuracy


parameters, costs,batchwise_accuracy = L_layer_model(train_X_f, train_y_f, layers_dims,epochs=1500,learning_rate=0.075, num_iterations = 2500, print_cost = True)
# a_file = open('parameters.pkl','wb')
# pk.dump(parameters,a_file)
# a_file.close()
a_file1 = open('costs.pkl','wb')
pk.dump(costs,a_file1)
a_file1.close()
a_file2 = open('batchwise_acc.pkl','wb')
pk.dump(batchwise_accuracy,a_file2)
a_file2.close()
# a_file = open('parameters.pkl','rb')
# parameters = pk.load(a_file)

accuracy = predictions(test_x_f,test_y_f,parameters)
print('accuracy : ',accuracy)
train_acc = predictions(trainX_acc,trainY_acc,parameters)
print('accuracy on training set : ',train_acc)
fig,ax1 = plt.subplots()
x = np.arange(1500)
ax1.plot(x,costs,label = 'cost')
ax1.set_xlabel('epochs')
ax1.legend()

fig2,ax2 = plt.subplots()
x2 = np.arange(len(batchwise_accuracy))
ax2.plot(x2,batchwise_accuracy,label = 'batch-wise accuracy')
# ax2.set_xlabel()

plt.show()

# accuracy :  0.4549     2layer
# accuracy on training set :  0.4422 2layer
# Cost after iteration 2900: 8.46382419476963e-08    3layer
# accuracy :  0.4284     3layer

# 4layer 784,300,100,30,10
# Cost after iteration 1499: 3.2484405563202684e-08
# accuracy :  0.9712
# accuracy on 10000 random examples on training set :  1.0



