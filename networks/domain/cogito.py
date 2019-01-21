import pathlib
import collections
import tensorflow as tf
from operator import mul
from functools import reduce
from abc import abstractmethod, ABCMeta






class base_network(metaclass=ABCMeta):
    def __init__(self):
        self._input_size = None
        self._output_size = None
        self._variables = []

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def variables(self):
        return self._variables

    def init(self, sess):
        for v in self._variables:
            sess.run(tf.variables_initializer(v))

    @abstractmethod
    def placeholders(self): pass

    @abstractmethod
    def configure(self): pass




class simple_network(base_network):
    '''
    This network is made from 3 layer, input, hidden, output
    '''
    def __init__(self, input_size, hidden_size, output_size):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self.configure()

    @property
    def _variables(self):
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


class base_optimizer(metaclass=ABCMeta):
    def __init__(self, name='base_optimizer'):
        self._name = name
        self.predictable = False
        self._variables = []

    @abstractmethod
    def configure(self, network): pass

    @abstractmethod
    def train(self): pass

    @abstractmethod
    def predict(self): pass


    @property
    def name(self):
        return self._name

    @property
    def variables(self):
        return self._variables

    def init(self, sess):
        for v in self._variables:
            sess.run(tf.variables_initializer(v))

    def save(self, sess):
        pass


class simple_bp(base_optimizer):
    """docstring for simple_optimizer"""
    def __init__(self, source, name='simple_bp'):
        self._source = source
        self._name = name
        self._variables = []

    def configure(self, network):
        raise self._source.shape == network.input_size
        self._input = network.input
        self._output = network.output
        X0 = tf.matmul(self._input, network.Wih)
        X1 = tf.tanh(tf.nn.batch_normalization(X0))
        self.output = tf.nn.softmax(tf.matmul(X1, network.Who))
        self._loss = tf.losses.softmax_cross_entropy(self._output, self.output)
        self._opt = tf.train.AdamOptimizer()
        self._variables = self._variables + self._opt.variables()

    def train(self, sess):
        input_data, output_data = self._source.__next__()
        sess.run(self._opt.minimize(self._loss),
            feed_dict={self._input: input_data, self._output: output_data})

    def predict(self. sess, test_source=False):
        if test_source and self._source.train:
            self._source.reset_source(train=False)
        input_data, output_data = self._source.__next__()
        sess.run(self.output, feed_dict={self._input: input_data})

class network_book:
    def __init__(self):
        self._network = None
        self._initialized = False

    def register(self, network):
        self._network = network()
        self._network.confiture()
        self._initialized = False
        for v in self._network.variables:
            self.set_variables(v)

    def extract_variables(self):
        for v in self._network.variables:
            yield v

    def reset_initialized_status(self):
        self._initialized = True

    @property
    def initialized(self):
        return self._initialized


class optimizer_book:
    def __init__(self):
        self._opt_list = []
        self._new_opt_list = []


    def register(self, optimizer, network):
        assert optimizer.name not in dir(self)
        opt = optimizer.configure(network)
        self._new_opt_list.append(opt)
        self._opt_list.append(opt)
        setattr(self, optimizer.name, opt)

    @property
    def names(self):
        return list(map(lambda obj:obj._name, self._opt_list))

    @property
    def predictable(self):
        predictable_iter = filter(lambda obj: obj.predictable, self._optself._opt_list)
        return list(map(lambda obj:obj._name, predictable_iter))

    def extract_from_name_list(self, name_list):
        for optname in name_list:
            yield getattr(self, optname)

    def reset_new_opt(self):
        self._new_opt_list = []
'''
<https://stackoverflow.com/questions/43068472/how-to-save-a-specific-variable-in-tensorflow>
'''
class cogito(object):
    """
    cogito is multi task training object
    cogito has one network and one or more optimizer
    """
    def __init__(self, save_path=None, save_name='cgt.ckpk'):
        if save_path is None:
            self._save_path = pathlib.Path.cwd() / save_name
        else:
            self._save_path = pathlib.Path(save_path) / save_name
        self._opt_book = optimizer_book()
        self._net_book = network_book()

    def set_networks(self, network):
        self._net_book.register(network)

    def set_optimizer(self, optimizer, source):
        if not isinstance(optimizer, base_optimizer):
            raise TypeError("Expected object of type base_optimizer, got {}".
                    format(type(optimizer).__name__))
        self._opt_book.register(optimizer(source()), self.network)

    def extract_new_variables(self):
        yield from self._opt_book.

    def train(self, iterate_number=10000, chains=self.optimizers):
        with tf.sesson as sess:
            # initialize
            if self._path.exists():
                self._saver.restore(sess, self._path)

            map(lambda v:sess.run(tf.variables_initializer(v)), self._used_variables)
            else:
                self._network.init(sess)
                map(lambda opt: opt.init(sess), self._opt_book.extract_from_name_list(chains))

            # train
            for i in range(iterate_number):
                for opt in self._opt_book.extract_from_name_list(chains):
                    opt.train(sess)
            
            # save
            self._saver = tf.train.Saver()
            self._saver.save(sess, self._save_path)


    def predict(self):
        pass



    @property
    def optimizers(self):
        return self._book.names

    def set_variables(variable):
        self._new_variables.append(variable)


########## source
class base_source(metaclass=ABCMeta):
    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self): pass

    @abstractmethod
    def shape(self): pass




class mnist(base_source):
    def __init__(self, train=True, melt=False):
        self._train = train
        self._melt = melt

        krs_mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = krs_mnist.load_data()
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

        self.reset_source()

    def __next__(self):
        x = self._x_train[self._cnt]
        y = self._y_train[self._cnt]
        if self._melt:
            x = x.reshape(self.shape)
        self._update_cnt()
        return x, y

    def _update_cnt(self):
        self._cnt = self._cnt + 1
        if self._cnt == self._len:
            self._cnt = 0

    def reset_source(self, train=True):
        if self._train:
            self._len = self._x_train.shape[0]
        else:
            self._len = self._x_test.shape[0]
        self._cnt = 0

    @property
    def shape(self):
        shape = self._x_train.shape[1:]
        if self._melt:
            shape = (reduce(mul, shape))
        return shape

def test

def main():
    network_size = {'input_size': 28 * 28, 
                 'hidden_size': 1000,
                 'output_size': 10}
    cgt = cogito()
    cgt.set_networks(simple_network(**network_size))
    cgt.set_optimizer(simple_bp(mnist(melt=True)))
    cgt.set_optimizer(simple_ae(mnist(melt=True)))
    cgt.train(iterate_number=10, chains=cgt.optimizers)
    cgt.predict(iterate_number=chains, chains=cgt.predictable)

if __name__ == '__main__':
    main()