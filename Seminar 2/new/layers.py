import numpy as np

class SGD:
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self, grads, learning_rate):
        for i in range(len(grads)):
            self.parameters[i] -= learning_rate * grads[i]

class Adam:
    def __init__(self, parameters, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(parameters[i]) for i in range(len(parameters))]
        self.v = [np.zeros_like(parameters[i]) for i in range(len(parameters))]

    def step(self, grads, learning_rate):
        self.t += 1
        for i in range(len(grads)):
            self.m[i] *= self.beta1
            self.m[i] += (1 - self.beta1) * grads[i]
            self.v[i] *= self.beta2
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2)
            self.parameters[i] -= learning_rate * self.m[i] / (1 - self.beta1 ** self.t) / (np.sqrt(self.v[i] / (1 - self.beta2 ** self.t)) + self.eps)

class Linear:
    def __init__(self, input_size, output_size):
        '''
        Creates weights and biases for linear layer.
        Dimention of inputs is *input_size*, of output: *output_size*.
        '''
        #### YOUR CODE HERE
        #### Create weights, initialize them with samples from N(0, 0.1).
        self.W = np.random.randn(input_size, output_size) * (2 / input_size)
        self.b = np.zeros(output_size)

        self.optimizer = Adam([self.W, self.b])

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, input_size).
        Returns output of size (N, output_size).
        Hint: You may need to store X for backward pass
        '''
        self.X = X
        return X.dot(self.W)+self.b

    def backward(self, dLdy):
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        self.dLdW = self.X.T.dot(dLdy)
        self.dLdb = dLdy.sum(0)
        self.dLdx = dLdy.dot(self.W.T)
        return self.dLdx

    def step(self, learning_rate):
        '''
        1. Apply gradient dLdw to network:
        w <- w - learning_rate*dLdw
        '''

        self.optimizer.step([self.dLdW, self.dLdb], learning_rate)
        # self.W = self.W - learning_rate * self.dLdW
        # self.b = self.b - learning_rate * self.dLdb
        
class Sigmoid:
    def __init__(self):
        pass
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        self.s = 1./(1+np.exp(-X))
        return self.s
    
    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        return self.s*(1-self.s)*dLdy
    
    def step(self, learning_rate):
        pass

class ReLU:
    def __init__(self):
        pass
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        #### YOUR CODE HERE
        #### Apply layer to input
        self.X = X
        return X.clip(min=0)
    
    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        #### YOUR CODE HERE
        return dLdy * (np.heaviside(self.X, 0))
    
    def step(self, learning_rate):
        pass

class ELU:
    def __init__(self):
        pass
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        #### YOUR CODE HERE
        #### Apply layer to input
        self.alpha = 1.0
        self.X = X
        pos = X.clip(min=0)
        self.neg = X.clip(max=0)
        return pos + self.alpha * (np.exp(self.neg) - 1)
    
    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        #### YOUR CODE HERE
        pos = np.heaviside(self.X, 0)
        return dLdy * (pos + self.alpha * np.exp(self.neg))
    
    def step(self, learning_rate):
        pass

class NLLLoss:
    def __init__(self):
        '''
        Applies Softmax operation to inputs and computes NLL loss
        '''
        pass
    
    def forward(self, X, y):
        '''
        Passes objects through this layer.
        X is np.array of size (N, C), where C is the number of classes
        y is np.array of size (N), contains correct labels
        '''
        self.p = np.exp(X)
        self.p /= self.p.sum(1, keepdims=True)
        self.y = np.zeros((X.shape[0], X.shape[1]))
        self.y[np.arange(X.shape[0]), y] = 1
        self.X = X
        return -(np.log(self.p)*self.y).sum(1).mean(0)
    
    def backward(self):
        '''
        Note that here dLdy = 1 since L = y
        1. Compute dLdx
        2. Return dLdx
        '''
        return (self.p - self.y) / self.X.shape[0]

class NeuralNetwork:
    def __init__(self, modules):
        '''
        Constructs network with *modules* as its layers
        '''
        #### YOUR CODE HERE
        self.modules = modules
    
    def forward(self, X):
        #### YOUR CODE HERE
        #### Apply layers to input
        for m in self.modules:
            X = m.forward(X)
        
        return X
    
    def backward(self, dLdy):
        '''
        dLdy here is a gradient from loss function
        '''
        #### YOUR CODE HERE
        for m in self.modules[::-1]:
            dLdy = m.backward(dLdy)
        
    
    def step(self, learning_rate):
        for m in self.modules:
            m.step(learning_rate)