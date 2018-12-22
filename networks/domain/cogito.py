import tensorflow as tf
from abc import ABCMeta, abstractmethod

class network_factory(metaclass=ABCMeta):
	"""docstring for network"""
	def __init__(self, arg):
		self.arg = arg


class network(metaclass=ABCMeta):
	"""docstring for network
	"""
	def __init__(self, input_size, output_size, hidden_size):
		self._input_size = input_size
		self._output_size = output_size
		self._hidden_size = hidden_size
		self.confiture()

	@property
	def input_size():
		return self._input_size

	@property
	def output_size():
		return self._output_size

	@property
	def hidden_size():
		return self._hidden_size


	@abstractmethod
	def confiture(self):
		pass
		
	
class simple_network(network):
	def __init__(self, **arg):
		super(simple_network, self).__init__(**arg)

	def confiture(self):
		self.input = tf.placeholder(tf.float32, [self._input_size])
		self.output = tf.placeholder(tf.float32, [self._output_size])
		self.hidden = tf.placeholder(tf.float32, [self._hidden_size])

		self.Wih = tf.Variable([self._hidden_size, self._input_size])
		self.Wio = tf.Variable([self._output_size, self._input_size])
		self.Whi = tf.Variable([self._input_size, self._hidden_size])
		self.Whh = tf.Variable([self._hidden_size, self._hidden_size])
		self.Who = tf.Variable([self._output_size, self._hidden_size])
		self.Woi = tf.Variable([self._input_size, self._output_size])
		self.Woh = tf.Variable([self._hidden_size, self._output_size])


class 





class basecogito(metaclass=ABCMeta):
	"""docstring for basecogito"""
	def __init__(self, arg):
		super(basecogito, self).__init__()
		self.arg = arg

	@abstractmethod
	def add_node(self):
		pass

	@abstractmethod
	def delite_node(self):
		pass

	@abstractmethod
	def add_ops(self):
		pass




