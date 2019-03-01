import pathlib
import collections
import numpy as np
import tensorflow as tf
from operator import mul
from functools import reduce
from abc import abstractmethod, ABCMeta

class base_network(metaclass=ABCMeta):
    def __init__(self):
        self._input_size = None
        self._output_size = None
        self._variables = []
        self._verbose = 0

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def variables(self):
        return self._variables

    @abstractmethod
    def placeholders(self): pass

    @abstractmethod
    def configure(self, verbose=0): pass




class simple_network(base_network):
    '''
    This network is made from 3 layer, input, hidden, output
    '''
    def __init__(self, input_size, hidden_size, output_size):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._verbose = 0

    @property
    def _variables(self):
        variables = [self.Wih, self.Wio, self.Whi,
            self.Whh, self.Who, self.Woi, self.Woh]
        return variables

    @property
    def placeholders(self):
        return [self.input, self.hidden, self.output]

    def configure(self, verbose=0):
        self._verbose = verbose
        self.input = tf.placeholder(tf.float32, [self._input_size], name='input')
        self.hidden = tf.placeholder(tf.float32, [self._hidden_size], name='hidden')
        self.output = tf.placeholder(tf.float32, [self._output_size], name='output')

#        self.Wih = tf.Variable([self._hidden_size, self._input_size], name='Wih')
#        self.Wio = tf.Variable([self._output_size, self._input_size], name='Wio')
#        self.Whi = tf.Variable([self._input_size, self._hidden_size], name='Whi')
#        self.Whh = tf.Variable([self._hidden_size, self._hidden_size], name='Whh')
#        self.Who = tf.Variable([self._output_size, self._hidden_size], name='Who')
#        self.Woi = tf.Variable([self._input_size, self._output_size], name='Woi')
#        self.Woh = tf.Variable([self._hidden_size, self._output_size], name='Woh')

        self.Wih = tf.get_variable('Wih', [self._hidden_size, self._input_size])
        self.Wio = tf.get_variable('Wio', [self._output_size, self._input_size])
        self.Whi = tf.get_variable('Whi', [self._input_size, self._hidden_size])
        self.Whh = tf.get_variable('Whh', [self._hidden_size, self._hidden_size])
        self.Who = tf.get_variable('Who', [self._output_size, self._hidden_size])
        self.Woi = tf.get_variable('Woi', [self._input_size, self._output_size])
        self.Woh = tf.get_variable('Woh', [self._hidden_size, self._output_size])

class base_optimizer(metaclass=ABCMeta):
    def __init__(self, name='base_optimizer', verbose=0):
        self._name = name
        self._variables = []
        self._verbose = verbose

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

#    @classmethod
#    def __subclasshook__(Class, Subclass):
#        if Class is Renderer:
#            attributes = collections.ChainMap(*(Superclass.__dict__
#                    for Superclass in Subclass.__mro__))
#            methods = ("header", "paragraph", "footer")
#            if all(method in attributes for method in methods):
#                return True
#        return NotImplemented



class simple_bp(base_optimizer):
    """docstring for simple_optimizer"""
    def __init__(self, source, name='simple_bp'):
        self._source = source
        self._name = name
        self._variables = []
        self._verbose = 0

    def configure(self, network, verbose=0):
        self._verbose = verbose
        assert self._source.shape == network.input_size
        self._input = network.input
        self._output = network.output
        if self._verbose >= 1:
            print('matmul check for X0')
            print('former shape: {shape}'.format(shape=self._input.shape))
            print('latter shape: {shape}'.format(shape=network.Wih.shape))
        X0 = tf.matmul(network.Wih, tf.reshape(self._input, [-1, 1]))
        X1 = tf.tanh(tf.nn.batch_normalization(X0, 0, 1, 0, 1, 1e-8))
        if self._verbose >= 1:
            print('matmul check for self.output')
            print('former shape: {shape}'.format(shape=network.Who.shape))
            print('latter shape: {shape}'.format(shape=X1.shape))
        self.output = tf.reshape(tf.nn.softmax(tf.matmul(network.Who, X1)), self._output.shape)
        self._loss = tf.losses.softmax_cross_entropy(self._output, self.output)
        self._opt = tf.train.AdamOptimizer()
        self._objective_function = self._opt.minimize(self._loss)
        self._variables = self._variables + self._opt.variables()

    def train(self, sess):
        input_data, output_data = self._source.__next__()
        sess.run(self._objective_function,
            feed_dict={self._input: input_data, self._output: output_data})
        if self._verbose >= 1:
            loss = sess.run(self._loss,
                feed_dict={self._input: input_data, self._output: output_data})
            print('{name} loss: {loss}'.format(name=self._name, loss=loss))

    def predict(self, sess, test_source=False):
        if test_source and self._source.train:
            self._source.reset_source(train=False)
        input_data, output_data = self._source.__next__()
        sess.run(self.output, feed_dict={self._input: input_data})

class network_catalog:
    def __init__(self, verbose=0):
        self.network = None
        self._initialized = True
        self._verbose = verbose

    def register(self, network):
        self.network = network
        self.network.configure(self._verbose)
#        self.network.configure()
        self._initialized = False
#        for v in self._network.variables:
#            self.set_variables(v)

    def extract_variables(self, new=False):
        if new and (not self._initialized):
            self._initialized = True
            yield from self._extract_variables()
        # if new and self._initialized:
        #     yield StopIteration
        if not new:
            yield from self._extract_variables()

    def _extract_variables(self):
        for v in self.network.variables:
            yield v

class optimizer_catalog:
    def __init__(self, verbose=0):
        self._opt_list = []
        self._new_opt_list = []
        self._verbose = verbose

    def register(self, optimizer, network):
        assert optimizer.name not in dir(self)
        optimizer.configure(network, self._verbose)
        self._new_opt_list.append(optimizer)
        self._opt_list.append(optimizer)
        setattr(self, optimizer.name, optimizer)

    @property
    def names(self):
        return list(map(lambda obj:obj._name, self._opt_list))

    @property
    def predictable(self):
        predictable_iter = filter(lambda obj: hasattr(obj, 'predict'), self._optself._opt_list)
        return list(map(lambda obj:obj._name, predictable_iter))

    def extract_from_name_list(self, name_list):
        for optname in name_list:
            yield getattr(self, optname)

    def extract_variables(self, new=False):
        if new:
            yield from self._extract_new_variables()
        else:
            yield from self._extract_all_variables()

    def _extract_new_variables(self):
        if self._verbose >= 1:
            print('in optimizer_catalog._extract_new_variables(self)')
            print('check optimizer_catalog._new_opt_list')
            print(self._new_opt_list)
        while len(self._new_opt_list) != 0:
            opt = self._new_opt_list.pop(0)
            for v in opt.variables:
                yield v

    def _extract_all_variables(self):
        for opt in self._opt_list:
            for v in opt.variables:
                yield v

'''
<https://stackoverflow.com/questions/43068472/how-to-save-a-specific-variable-in-tensorflow>
'''
class cogito(object):
    """
    cogito is multi task training object
    cogito has one network and one or more optimizer
    """
    def __init__(self, save_path=None, save_name='cgt.ckpk', initialize=False, verbose=0):
        self._save_name = save_name
        self._verbose = verbose

        self._saver = None
        if save_path is None:
            self._save_path = pathlib.Path.cwd()
        else:
            self._save_path = pathlib.Path(save_path)
        if initialize:
            for trg in self._save_path.glob(self._save_name + '*'):
                trg.unlink()

        self._opt_catalog = optimizer_catalog(verbose=verbose)
        self._net_catalog = network_catalog(verbose=verbose)

    def set_networks(self, network):
        self._net_catalog.register(network)

    def set_optimizer(self, optimizer, source):
        opt = optimizer(source)
        if not isinstance(opt, base_optimizer):
            raise TypeError("Expected object of type base_optimizer, got {}".
                            format(type(optimizer).__name__))
        self._opt_catalog.register(opt, self._net_catalog.network)

    def _extract_variables(self, new=False):
        yield from self._net_catalog.extract_variables(new=new)
        yield from self._opt_catalog.extract_variables(new=new)


    def train(self, iterate_number=10000, chains=[]):

        if len(chains) == 0:
            chains = self.optimizers

        with tf.Session() as sess:
            # initialize
            if self._save_path.exists():
                self._saver.restore(sess, self._save_path)

            sess.run(tf.variables_initializer(self._extract_variables(new=True)))
            # for v in self._extract_variables(new=True):
            #     sess.run(tf.variables_initializer(v))

            # train
            for i in range(iterate_number):
                if self._verbose >= 3:
                    print(i)
                for opt in self._opt_catalog.extract_from_name_list(chains):
                    opt.train(sess)

            # save
            self._saver = tf.train.Saver(list(self._extract_variables()))
            # self._saver.save(sess, self._save_path)
            self._saver.save(sess, str(self._save_path))


    def predict(self, iterate_number=10000, chains=[]):
        with tf.Session() as sess:
            # initialize
            if self._save_path.exists():
                self._saver.restore(sess, self._save_path)

            sess.run(tf.variables_initializer(self._extract_variables(new=True)))
            # for v in self._extract_variables(new=True):
            #     sess.run(tf.variables_initializer(v))

            # train
            for i in range(iterate_number):
                if self._verbose >= 3:
                    print(i)
                for opt in self._opt_catalog.extract_from_name_list(chains):
                    opt.train(sess)

            # save
            self._saver = tf.train.Saver(list(self._extract_variables()))
            # self._saver.save(sess, self._save_path)
            self._saver.save(sess, str(self._save_path))


    @property
    def optimizers(self):
        return self._opt_catalog.names

########## source
class base_source(metaclass=ABCMeta):
    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self): pass

    @abstractmethod
    def shape(self): pass




class mnist(base_source):
    def __init__(self, train=True, melt=False, dummy=True):
        self._train = train
        self._melt = melt
        self._dummy = dummy

        krs_mnist = tf.keras.datasets.mnist
        xy_train, xy_test = krs_mnist.load_data()
        self._x_train, self._y_train = xy_train
        self._x_test, self._y_test = xy_test

        self.reset_source()

    def __next__(self):
        x = self._x_train[self._cnt]
        y = self._y_train[self._cnt]
        if self._melt:
            x = x.reshape(self.shape)
        if self._dummy:
            assert isinstance(y, (int, np.integer))
            assert y >= 0
            assert y <= 9
            y = np.array([0] * y + [1] + [0] * (9 - y))

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

def test():
    ## source
    source = mnist(melt=True, dummy=False)
    x, y = next(source)
    isinstance(y, np.uint8)
    tf.reset_default_graph()
    network_size = {'input_size': 28 * 28,
                 'hidden_size': 1000,
                 'output_size': 10}
    cgt = cogito(verbose=3, initialize=True)
    cgt.set_networks(simple_network(**network_size))
    cgt.set_optimizer(simple_bp, mnist(melt=True))
#    cgt.set_optimizer(simple_ae(mnist(melt=True)))
    print(cgt.optimizers)
    # print(list(cgt._extract_variables()))
    for v in cgt._extract_variables():
        print(v.name)
    cgt.train(iterate_number=100, chains=cgt.optimizers)
    cgt.train(iterate_number=100, chains=cgt.optimizers)

    print(cgt.predict('simple_bp'))


def main():
    network_size = {'input_size': 28 * 28,
                    'hidden_size': 1000,
                    'output_size': 10}
    cgt = cogito()
    cgt.set_networks(simple_network(**network_size))
    cgt.set_optimizer(simple_bp, mnist(melt=True))
    cgt.set_optimizer(simple_ae(mnist(melt=True)))
    cgt.train(iterate_number=10, chains=cgt.optimizers)
    cgt.predict(iterate_number=chains, chains=cgt.predictable)

if __name__ == '__main__':
    main()