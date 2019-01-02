import Qtrac
import tensorflow as tf
from abc import ABCMeta, abstractmethod

class network_factory(metaclass=ABCMeta):
    """docstring for network"""
    def __init__(self):
        pass

    def create_network():




nwo = network(input_size, output_size, hidden_size)
nwo.



@Qtrac.has_methods('variables', 'placeholders', 'confiture')
class base_network(metaclass=abc.ABCMeta): pass

@Qtrac.has_methods('configure', 'train')
class base_optimizer(metaclass=abc.ABCMeta): pass   

@Qtrac.has_methods('shape', 'draw')
class base_source(metaclass=abc.ABCMeta): pass   


    
class simple_network(base_network):
    def __init__(self, input_size, hidden_size, output_size):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        super(simple_network, self).__init__()

    @property
    def variables(self):
        variables = [self.Wih, self.Wio, self.Whi,
            self.Whh, self.Who, self.Woi, self.Woh]
        return variables
    
    @property
    def placeholders(self):
        return [self.input, self.hidden, self.output]

    def confiture(self):
        self.input = tf.placeholder(tf.float32, [self._input_size], name='input')
        self.hidden = tf.placeholder(tf.float32, [self._hidden_size], name='hidden')
        self.output = tf.placeholder(tf.float32, [self._output_size], name='output')

        self.Wih = tf.Variable([self._hidden_size, self._input_size], name='Wih')
        self.Wio = tf.Variable([self._output_size, self._input_size], name='Wio')
        self.Whi = tf.Variable([self._input_size, self._hidden_size], name='Whi')
        self.Whh = tf.Variable([self._hidden_size, self._hidden_size], name='Whh')
        self.Who = tf.Variable([self._output_size, self._hidden_size], name='Who')
        self.Woi = tf.Variable([self._input_size, self._output_size], name='Woi')
        self.Woh = tf.Variable([self._hidden_size, self._output_size], name='Woh')

    def save(self):
        pass


class optimizer_book:
    def __init__(self):
        self._opt_list = []

    def register(self, optimizer, network):
        opt = optimizer(network)
        self._opt_list.append(opt)
        return opt

    @property
    def names(self):
        return list(map(lambda obj:obj._name, self._opt_list))



class simple_bp(base_optimizer):
    """docstring for simple_optimizer"""
    def __init__(self, source):
        self._source = source

    def configure(self, network):
        X0 = tf.matmul(network.input, network.Wih)
        X1 = tf.tanh(tf.nn.batch_normalization(X0))
        output = tf.nn.softmax(tf.matmul(X1, network.Who))
        self._loss = tf.losses.softmax_cross_entropy(network.output, output)

    def train(self):
        return self




class cogito(object):
    """docstring for cogito"""
    def __init__(self, arg):
        super(cogito, self).__init__()
        self.arg = arg
        
    def set_networks(self, network):
        self._network = network()
        self._network.confiture()

    def set_optimizer(self, optimizer, source):
        if not isinstance(renderer, base_optimizer):
            raise TypeError("Expected object of type base_optimizer, got {}".
                    format(type(renderer).__name__))
        optimizer(source())

@Qtrac.has_methods('shape', '__next__')
class base_source(metaclass=abc.ABCMeta): pass   

class mnist(base_source):
    def __init__(self):
        krs_mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = krs_mnist.load_data()
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self.len = x_train.shape[0]
        
        pass

    @property
    def shape(self):
        pass

    def __next__(self):
        return self

data = mnist.load_data()

def main():
    network_size = {'input_size': 28 * 28, 
                 'hidden_size': 1000,
                 'output_size': 10}
    cgt = cogito()
    cgt.set_networks(simple_network(**network_size))
    cgt.set_optimizer(simple_bp(mnist()))
    cgt.set_optimizer(simple_ae(mnist()))
    cgt.train(epoch=10, chains=cgt.optimizers)
    

if __name__ == '__main__':
    main()