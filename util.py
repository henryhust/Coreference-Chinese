from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import codecs
import collections
import shutil
import sys

import numpy as np
import tensorflow as tf
import pyhocon

import independent0

def get_model(config):
    if config['model_type'] == 'independent':
        if config["use_pos"]:
            return independent.CorefModel_pos(config)     # bert + pos
        else:
            return independent0.CorefModel(config)         # bert
    elif config['model_type'] == "independent-xlnet":
        import independent_xlnet
        if not config["use_pos"]:

            return independent_xlnet.CorefModel(config)         # xlnet
    else:
        raise NotImplementedError('Undefined model type')


def initialize_from_env(eval_test=False):
  if "GPU" in os.environ:
    set_gpus(int(os.environ["GPU"]))

  name = sys.argv[1]
  print("Running experiment: {}".format(name))

  if eval_test:
    config = pyhocon.ConfigFactory.parse_file("test.experiments.conf")[name]
  else:
    config = pyhocon.ConfigFactory.parse_file("config/experiments_tagging.conf")[name]

  if "chinese" in config["train_path"]:
    """训练数据遵循命名规则xx.chinese.128.jsonlines"""
    print("路径创建成功")
    config["log_dir"] = mkdirs(config["save_path"])
  else:
    config["log_dir"] = mkdirs(os.path.join(config["log_root"], name))
  print(pyhocon.HOCONConverter.convert(config, "hocon"))
  return config


def copy_checkpoint(source, target):
  for ext in (".index", ".data-00000-of-00001"):
    shutil.copyfile(source + ext, target + ext)

def make_summary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])


def flatten(l):
  return [item for sublist in l for item in sublist]


def set_gpus(*gpus):
  # pass
  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
  print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

def mkdirs(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise
  return path

def load_char_dict(char_vocab_path):
  vocab = [u"<unk>"]
  with codecs.open(char_vocab_path, encoding="utf-8") as f:
    vocab.extend(l.strip() for l in f.readlines())
  char_dict = collections.defaultdict(int)
  char_dict.update({c:i for i, c in enumerate(vocab)})
  return char_dict

def maybe_divide(x, y):
  return 0 if y == 0 else x / float(y)

def projection(inputs, output_size, initializer=tf.truncated_normal_initializer(stddev=0.02)):
  return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)


def highway(inputs, num_layers, dropout):
  for i in range(num_layers):
    with tf.variable_scope("highway_{}".format(i)):
      j, f = tf.split(projection(inputs, 2 * shape(inputs, -1)), 2, -1)
      f = tf.sigmoid(f)
      j = tf.nn.relu(j)
      if dropout is not None:
        j = tf.nn.dropout(j, dropout)
      inputs = f * j + (1 - f) * inputs
  return inputs

def shape(x, dim):
  return x.get_shape()[dim].value or tf.shape(x)[dim]

def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=tf.truncated_normal_initializer(stddev=0.02), hidden_initializer=tf.truncated_normal_initializer(stddev=0.02)):
  if len(inputs.get_shape()) > 3:
    raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

  if len(inputs.get_shape()) == 3:
    batch_size = shape(inputs, 0)
    seqlen = shape(inputs, 1)
    emb_size = shape(inputs, 2)
    current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
  else:
    current_inputs = inputs

  for i in range(num_hidden_layers):
    hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size], initializer=hidden_initializer)
    hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size], initializer=tf.zeros_initializer())
    current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

    if dropout is not None:
      current_outputs = tf.nn.dropout(current_outputs, dropout)
    current_inputs = current_outputs

  output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
  output_bias = tf.get_variable("output_bias", [output_size], initializer=tf.zeros_initializer())
  outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

  if len(inputs.get_shape()) == 3:
    outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
  return outputs

def linear(inputs, output_size):
  if len(inputs.get_shape()) == 3:
    batch_size = shape(inputs, 0)
    seqlen = shape(inputs, 1)
    emb_size = shape(inputs, 2)
    current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
  else:
    current_inputs = inputs
  hidden_weights = tf.get_variable("linear_w", [shape(current_inputs, 1), output_size])
  hidden_bias = tf.get_variable("bias", [output_size])
  current_outputs = tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias)
  return current_outputs

def cnn(inputs, filter_sizes, num_filters):
  num_words = shape(inputs, 0)
  num_chars = shape(inputs, 1)
  input_size = shape(inputs, 2)
  outputs = []
  for i, filter_size in enumerate(filter_sizes):
    with tf.variable_scope("conv_{}".format(i)):
      w = tf.get_variable("w", [filter_size, input_size, num_filters])
      b = tf.get_variable("b", [num_filters])
    conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
    h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
    pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
    outputs.append(pooled)
  return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]

def batch_gather(emb, indices):
  batch_size = shape(emb, 0)
  seqlen = shape(emb, 1)
  if len(emb.get_shape()) > 2:
    emb_size = shape(emb, 2)
  else:
    emb_size = 1
  flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
  offset = tf.expand_dims(tf.range(batch_size) * seqlen, 1)  # [batch_size, 1]
  gathered = tf.gather(flattened_emb, indices + offset) # [batch_size, num_indices, emb]
  if len(emb.get_shape()) == 2:
    gathered = tf.squeeze(gathered, 2) # [batch_size, num_indices]
  return gathered

def biLSTM_layer(model_inputs, lstm_dim, lengths, name=None):
    """
    :param lstm_inputs: [batch_size, num_steps, emb_size]
    :return: [batch_size, num_steps, 2*lstm_dim]
    """
    from tensorflow.contrib.layers.python.layers import initializers
    with tf.variable_scope("char_BiLSTM" if not name else name):
        lstm_cell = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
                lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
                    lstm_dim,
                    use_peepholes=True,
                    initializer=initializers.xavier_initializer(),
                    state_is_tuple=True)
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell["forward"],
            lstm_cell["backward"],
            model_inputs,
            dtype=tf.float32,
            sequence_length=lengths)
    return tf.concat(outputs, axis=2)


def biGRU_layer(model_inputs, gru_dim, lengths, name=None):
    """
    :param lstm_inputs: [batch_size, num_steps, emb_size]
    :return: [batch_size, num_steps, 2*lstm_dim]
    """
    from tensorflow.contrib.layers.python.layers import initializers
    with tf.variable_scope("char_BiGRU" if not name else name):
        gru_cell = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
              gru_cell[direction] = tf.contrib.rnn.GRUCell(gru_dim)
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
            gru_cell["forward"],
            gru_cell["backward"],
            model_inputs,
            dtype=tf.float32,
            sequence_length=lengths)
    return tf.concat(outputs, axis=2)

def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    # if isinstance(inputs, tuple):
    #     # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
    #     inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    initializer = tf.random_normal_initializer(stddev=0.1)

    # Trainable parameters
    w_omega = tf.get_variable(name="w_omega", shape=[hidden_size, attention_size], initializer=initializer)
    b_omega = tf.get_variable(name="b_omega", shape=[attention_size], initializer=initializer)
    u_omega = tf.get_variable(name="u_omega", shape=[attention_size], initializer=initializer)

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


class RetrievalEvaluator(object):
  def __init__(self):
    self._num_correct = 0
    self._num_gold = 0
    self._num_predicted = 0

  def update(self, gold_set, predicted_set):
    self._num_correct += len(gold_set & predicted_set)
    self._num_gold += len(gold_set)
    self._num_predicted += len(predicted_set)

  def recall(self):
    return maybe_divide(self._num_correct, self._num_gold)

  def precision(self):
    return maybe_divide(self._num_correct, self._num_predicted)

  def metrics(self):
    recall = self.recall()
    precision = self.precision()
    f1 = maybe_divide(2 * recall * precision, precision + recall)
    return recall, precision, f1

class EmbeddingDictionary(object):
  def __init__(self, info, normalize=True, maybe_cache=None):
    self._size = info["size"]
    self._normalize = normalize
    self._path = info["path"]
    self.vocab = []
    if maybe_cache is not None and maybe_cache._path == self._path:
      assert self._size == maybe_cache._size
      self._embeddings = maybe_cache._embeddings
    else:
      self._embeddings, self._embedding_param = self.load_embedding_dict(self._path)
    self.char2ids = collections.defaultdict(int)
    self.char2ids.update({char: i + 1 for i, char in enumerate(self.vocab)})
    self.char2ids.update({"UNK": 0})
    if self._normalize:
      self._embedding_param = self.normalize(self._embedding_param)

  @property
  def size(self):
    return self._size

  def load_embedding_dict(self, path):
    print("Loading word embeddings from {}...".format(path))
    default_embedding = np.random.uniform(-0.25, 0.25, self.size)
    embedding_dict = collections.defaultdict(lambda:default_embedding)
    embedding_param = []
    if len(path) > 0:
      vocab_size = None
      with open(path) as f:
        for i, line in enumerate(f.readlines()):
          word_end = line.find(" ")
          word = line[:word_end]
          self.vocab.append(word)
          embedding = np.fromstring(line[word_end + 1:], np.float32, sep=" ")
          embedding_param.append(embedding)
          assert len(embedding) == self.size
          embedding_dict[word] = embedding
      if vocab_size is not None:
        assert vocab_size == len(embedding_dict)
      embedding_param.insert(0, [0] * 300)
      print("Done loading word embeddings.")
    return embedding_dict, embedding_param

  def __getitem__(self, key):
    embedding = self._embeddings[key]
    if self._normalize:
      embedding = self.normalize(embedding)
    return embedding

  def normalize(self, v):
    """求解矩阵范数"""
    norm = np.linalg.norm(v)
    if norm > 0:
      return v / norm
    else:
      return v

  def convert_tokens_to_ids(self, sentence):
    sent_char_ids = [self.char2ids[char] for char in sentence]
    return sent_char_ids


class CustomLSTMCell(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units, batch_size, dropout):
    self._num_units = num_units
    self._dropout = dropout
    self._dropout_mask = tf.nn.dropout(tf.ones([batch_size, self.output_size]), dropout)
    self._initializer = self._block_orthonormal_initializer([self.output_size] * 3)
    initial_cell_state = tf.get_variable("lstm_initial_cell_state", [1, self.output_size])
    initial_hidden_state = tf.get_variable("lstm_initial_hidden_state", [1, self.output_size])
    self._initial_state = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_hidden_state)

  @property
  def state_size(self):
    return tf.contrib.rnn.LSTMStateTuple(self.output_size, self.output_size)

  @property
  def output_size(self):
    return self._num_units

  @property
  def initial_state(self):
    return self._initial_state

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "CustomLSTMCell"
      c, h = state
      h *= self._dropout_mask
      concat = projection(tf.concat([inputs, h], 1), 3 * self.output_size, initializer=self._initializer)
      i, j, o = tf.split(concat, num_or_size_splits=3, axis=1)
      i = tf.sigmoid(i)
      new_c = (1 - i) * c  + i * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)
      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      return new_h, new_state

  def _orthonormal_initializer(self, scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
      M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
      Q1, R1 = np.linalg.qr(M1)
      Q2, R2 = np.linalg.qr(M2)
      Q1 = Q1 * np.sign(np.diag(R1))
      Q2 = Q2 * np.sign(np.diag(R2))
      n_min = min(shape[0], shape[1])
      params = np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
      return params
    return _initializer

  def _block_orthonormal_initializer(self, output_sizes):
    def _initializer(shape, dtype=np.float32, partition_info=None):
      assert len(shape) == 2
      assert sum(output_sizes) == shape[1]
      initializer = self._orthonormal_initializer()
      params = np.concatenate([initializer([shape[0], o], dtype, partition_info) for o in output_sizes], 1)
      return params
    return _initializer
