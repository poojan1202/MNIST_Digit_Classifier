# MNIST CLASSIFIER

This project is focussed at classification of 0-9 handwritten digits using [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset, using Artificial Neural Network (ANN) and Convolutional Neural Network (CNN).

- The training dataset contains 60,000 handwritten digit images, each of 28x28 pixels.
    - File : train-images.idx3-ubyte
- The labels of these images are stored in seperate file
    - File : train-labels.idx1-ubyte
- The test dataset contains 10,000 handwritten digit images, each of 28x28 pixels.
    - File : t10k-images.idx3-ubyte
- The labels of these test images are stored in seperate file
    - File : t10k-labels.idx1-ubyte

These files are already present in the repository, but can still be downloaded from [here](http://yann.lecun.com/exdb/mnist/).


    

## 1. Artificial Neural Network
Performed with two different models
- 3-Layered Model
    - layers_dims = 
- 4-Layered Model
    - 4 layers, with 300,100,30 and 10 nodes in it
    - layers_dims = [784,300,100,30,10]

Various Parameters set while training the model


| Parameter | Value |
| -------- | -------- |
| learning rate     | 0.075     |
| epochs | 1000|
| Mini batch size | 128|

#### Activation Functions
- All hidden layers are applied with **Relu** Activation Function
    - $A = max(Z,0)$
- The last (output) layer is applied with **Softmax** Activation Function
    - $A_i = \dfrac{e^{z_i}}{\Sigma^K_{j=1}e^{z_j}}$
        - Here K represents the number of node in that layer (10 here)
- Here, $Z = W^TX + b$

### Results

#### 3-Layered Model
- Cost-Function
![](https://i.imgur.com/Z1lFGQd.png)
- Cost after 1000 epochs: 5.317678221962305e-08
- BatchWise Accuracy
![](https://i.imgur.com/twFRBPb.png)

- Accuracy of Trained Model on Training Dataset : **1.0**
- Accuracy of Trained Model on Test Dataset : **0.9712**

Trained model can be found inside Results folder
```python=
a_file = open('Results/ANN_4/parameters.pkl','rb')
parameters = pk.load(a_file)
```
Run above block of code to test on the already trained model

#### 4-Layered Model
- Cost-Function
- BatchWise Accuracy
- Accuracy of Trained Model on Training Dataset : ****
- Accuracy of Trained Model on Test Dataset : ****




## 2. Convolution Neural Network

## Dependencies

- idx2numpy
- numpy
- matplotlib
- pickle