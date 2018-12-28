import tensorflow as tf
from abc import ABCMeta, abstractmethod

class network_factory(metaclass=ABCMeta):
    """docstring for network"""
    def __init__(self):
        pass

    def create_network():




nwo = network(input_size, output_size, hidden_size)
nwo.






class base_network(metaclass=ABCMeta):
    """docstring for base_network
    """
    def __init__(self):
        self.confiture()

    @property
    @abstractmethod
    def variables(self):
        pass
    
    @property
    @abstractmethod
    def placeholders(self):
        pass

    @abstractmethod
    def confiture(self):
        pass
        
    
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

class base_optimizer(metaclass=ABCMeta):
    """docstring for base_optimizer"""
    def __init__(self, source):
        self._source = source

    @abstractmethod
    def configure(self, network):
        pass
    
    @abstactmethod
    def train(self):
        pass

class simple_bp(base_optimizer):
    """docstring for simple_optimizer"""
    def __init__(self, source):
        super(simple_bp, self).__init__(source)

    def configure(self, network):
        X0 = tf.matmul(network.input, network.Wih)
        X1 = tf.tanh(tf.nn.batch_normalization(X0)
        output = tf.nn.softmax(tf.matmul(X1, network.Who))
        self._loss = tf.losses.softmax_cross_entropy(network.output, output)

    def train(self):
        pass


class hi:
    pass


class cogito(object):
    """docstring for cogito"""
    def __init__(self, arg):
        super(cogito, self).__init__()
        self.arg = arg
        
    def add_node(self):
        pass



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