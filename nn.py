import random
import numpy as np
from tensor import Tensor

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Linear(Module):

    def __init__(self, fan_in, fan_out, bias=True, gain=1):
        """
        construct a linear layer
        we can disable b for BN for exemple
        gain is set by default to 1 -> gain is used to optimize init
        """
        self.weight = Tensor( ( np.random.randn(fan_in, fan_out) * (gain / np.sqrt(fan_in)) ) )
        self.bias = Tensor( np.random.randn(1, fan_out) ) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class ReLU(Module):
    def __init__(self):
        pass

    def __call__(self, x):
        return x.relu()
    
    def parameters(self):
        return []

class Softmax(Module):
    def __init__(self):
        pass

    def __call__(self, x):
        return x.softmax()
    
    def parameters(self):
        return []
    
class Tanh(Module):
    def __init__(self):
        pass

    def __call__(self, x):
        return x.tanh()
    
    def parameters(self):
        return []
    
class BN(Module):
    def __init__(self, dim, epsilon=1e-05):
        """
        without running mean / var
        """
        self.beta = Tensor(np.zeros((1, dim)))
        self.gamma = Tensor(np.ones((1, dim)))
        self.epsilon = epsilon

    def __call__(self, x):
        new_x = (x - np.mean(x.data)) / np.sqrt((np.var(x.data) + self.epsilon))
        out = self.gamma * new_x + self.beta
        return out
        
    def parameters(self):
        return [self.gamma] + [self.beta]

class SequentialNN(Module):

    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

