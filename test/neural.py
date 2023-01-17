import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


class Weights(np.ndarray):
    @classmethod
    def single_column(cls, arr):
        weights = Weights(len(arr), 1)
        weights[:, 0] = arr
        return weights

    def __new__(cls, r, c):
        weights = np.zeros((r, c), dtype = float).view(cls)
        return weights

    rows = property(lambda self: self.shape[0])
    cols = property(lambda self: self.shape[1])

    def randomize(self):
        self[:] = np.random.random(self.shape) * 2. - 1.
        return self

    def mutate(self, mutation_rate):
        mutate = np.random.random(self.shape) < mutation_rate
        rand = np.random.normal(size = self.shape) / 5
        np.add(self, rand, out = self, where = mutate)
        np.clip(self, -1., 1., out = self)

    def add_bias(self):
        weights = Weights(self.rows + 1, self.cols)
        weights[:-1, 0] = self[:, 0]
        weights[-1, 0] = 1
        # weights = np.append(self, [[1]], axis = 0).view(self.__class__)
        return weights

    def activate(self):
        self[:] = sigmoid(self)
        return self

    def crossover(self, partner):
        child = Weights(self.rows, self.cols)
        flat1, flat2 = self.flatten(), partner.flatten()
        mid = int(np.random.random() * len(flat1))
        child[:] = np.append(flat1[:mid], flat2[mid:]).reshape(self.shape)
        return child


class NeuralNet:
    def __init__(self, inputs, hidden, outputs):
        self.inputs, self.hidden, self.outputs = inputs, hidden, outputs

        # Weights from Input Nodes to Hidden Layer 1
        self.w_hi = Weights(hidden, inputs + 1).randomize()
        # Weights from Hidden Layer 1 Nodes to Hidden Layer 2
        self.w_hh = Weights(hidden, hidden + 1).randomize()
        # Weights from Hidden Layer 2 Nodes to Output
        self.w_oh = Weights(outputs, hidden + 1).randomize()

    def mutate(self, mutation_rate):
        self.w_hi.mutate(mutation_rate)
        self.w_hh.mutate(mutation_rate)
        self.w_oh.mutate(mutation_rate)

    def output(self, input_arr):
        data = Weights.single_column(input_arr).add_bias()

        data = self.w_hi.dot(data).activate().add_bias()
        data = self.w_hh.dot(data).activate().add_bias()
        data = self.w_oh.dot(data).activate()

        return data.flatten()

    def crossover(self, partner):
        child = NeuralNet(self.inputs, self.hidden, self.outputs)
        child.w_hi = self.w_hi.crossover(partner.w_hi)
        child.w_hh = self.w_hh.crossover(partner.w_hh)
        child.w_oh = self.w_oh.crossover(partner.w_oh)
        return child

    def copy(self):
        new = NeuralNet(self.inputs, self.hidden, self.outputs)
        new.w_hi = self.w_hi.copy()
        new.w_hh = self.w_hh.copy()
        new.w_oh = self.w_oh.copy()
        return new


if __name__ == '__main__':
    net1 = NeuralNet(5, 5, 5)
    net2 = NeuralNet(5, 5, 5)
    # print(net1.output([1, 2, 3, 4, 5]))
    print(net1.crossover(net2))
