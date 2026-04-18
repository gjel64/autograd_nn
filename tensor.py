import numpy as np
class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.grad = 0
        self.require_grad = False
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Tensor({self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            # y = x + z
            # dy / dx = 1
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            # y = x * z
            # dy / dx = z
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _op="@")

        def _backward():
            # y = x * z
            # dy / dx = z
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        # y = x ** z
        # dy / dx = z * (x ** (z-1))
        assert isinstance(other, (int, float))
        out = Tensor(self.data**other, _children=(self,), _op="**")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Tensor(np.exp(self.data), _children=(self,), _op="exp")

        def _backward():
            # y = e**x
            # dy / dx = e**x
            self.grad += np.exp(self.data) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), _children=(self,), _op="ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        out = Tensor(
            (self.exp() - (-self).exp()) / (self.exp() + (-self).exp()),
            _children=(self,),
            _op="tanh",
        )

        def _backward():
            # from wikipedia
            self.grad += (1 - (self.tanh() ** 2)) * out.grad

        out._backward = _backward

        return out

    def softmax(self):
        if len(self.data.shape) == 0:
          out = Tensor(1, _children=(self,), _op="softmax")

          def _backward():
            self.grad += 0 * out.grad
          out._backward = _backward

          return out

        else:
            shifted = self.data - np.max(self.data, axis=-1, keepdims=True)
            exps = np.exp(shifted)
            probs = exps / np.sum(exps, axis=-1, keepdims=True)
            
            out = Tensor(probs, _children=(self,), _op="softmax")
            
            def _backward():
                # Gradient of softmax: p_i * (dL/dy_i - sum_j(p_j * dL/dy_j))
                sum_p_grad = np.sum(probs * out.grad, axis=-1, keepdims=True)
                self.grad += probs * (out.grad - sum_p_grad)
                
            out._backward = _backward
            return out


    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1


    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

