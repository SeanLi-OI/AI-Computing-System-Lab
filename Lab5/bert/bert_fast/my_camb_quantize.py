from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import importer
from tensorflow.python.ops import variables
from tensorflow.core.protobuf import config_pb2


def create_node(op, name, inputs):
  new_node = node_def_pb2.NodeDef()
  new_node.op = op
  new_node.name = name
  for input_name in inputs:
    new_node.input.extend([input_name])
  return new_node


def create_constant_node(name, value, dtype, shape=None):
  node = create_node("Const", name, [])
  set_attr_dtype(node, "dtype", dtype)
  set_attr_tensor(node, "value", value, dtype, shape)
  return node


def copy_attr(node, key, attr_value):
  try:
    node.attr[key].CopyFrom(attr_value)
  except KeyError:
    pass


def set_attr_dtype(node, key, value):
  try:
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(type=value.as_datatype_enum))
  except KeyError:
    pass


def set_attr_shape(node, key, value):
  try:
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(shape=tensor_shape.as_shape(value).as_proto()))
  except KeyError:
    pass


def set_attr_tensor(node, key, value, dtype, shape=None):
  try:
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            value, dtype=dtype, shape=shape)))
  except KeyError:
    pass


def set_attr_string(node, key, value):
  try:
    node.attr[key].CopyFrom(attr_value_pb2.AttrValue(s=value))
  except KeyError:
    pass


def set_attr_int_list(node, key, value):
  list_value = attr_value_pb2.AttrValue.ListValue(i=value)
  try:
    node.attr[key].CopyFrom(attr_value_pb2.AttrValue(list=list_value))
  except KeyError:
    pass


def set_attr_bool(node, key, value):
  try:
    node.attr[key].CopyFrom(attr_value_pb2.AttrValue(b=value))
  except KeyError:
    pass


def set_attr_int(node, key, value):
  try:
    node.attr[key].CopyFrom(attr_value_pb2.AttrValue(i=value))
  except KeyError:
    pass


def set_attr_float(node, key, value):
  try:
    node.attr[key].CopyFrom(attr_value_pb2.AttrValue(f=value))
  except KeyError:
    pass

class QuantizeGraph(object):

  QUANTIZATION_TYPE_INT8  = "INT8"
  QUANTIZATION_TYPE_INT16 = "INT16"

  def __init__(self,
               input_graph_def,
               layer_num,
               output_node_names,
               quantization_output_node_names,
               scope_names,
               post_process_name,
               quantization_type=QUANTIZATION_TYPE_INT16,
               device_mode='clean',
               int_op_list=['FC', 'MatMul']):
    """Sets up the class to fix input graph.

    Args:
      input_graph_def: a instance of tf.GraphDef,
        the graph that will be converted to int graph.
      layer_num: num of encoder layers in Bert
      output_node_names: A list, contain all names of output nodes.
      scope_names: scope names of tf.dense and tf.batch_matmul in Bert
      post_process_name: scope names of tf.dense in post process of Bert
      quantization_type: A string, must be any of [QUANTIZATION_TYPE_INT8, QUANTIZATION_TYPE_INT16]
      device_node: A string, must be in ['clean', 'mlu',  'origin'].
        clean - delete all operation's devece.
        mlu - set all operation on mlu device.
        origin - don't change any operation's device.
      int_op_list: A list of string, to choose which layer to int.
    """
    self.input_max = dict()
    self.input_min = dict()
    self.const_dict = dict()
    self.layer_num = layer_num
    self.output_node_names = output_node_names
    self.quantization_output_node_names = quantization_output_node_names
    self.scope_names = scope_names
    self.post_process_name = post_process_name
    self.int_op_list = int_op_list

    if quantization_type == QuantizeGraph.QUANTIZATION_TYPE_INT16:
        self.INT_MAX = 32767
        self.INT_MIN = -32767
        self.POS_RANGE_ABS = 32
        self.NP_INT_TYPE = np.int16
        self.DTYPE_INT_TYPE = dtypes.int16
        self.FAKEQUANTSCALEINTGEN_OP_NAME = "FakeQuantScaleInt16Gen"
    else:
      raise Exception("Quantization type {} is not supported".format(quantization_type))

    if input_graph_def:
      self.input_graph = self.build_fake_int_graph(input_graph_def)
    else:
      raise ValueError("Input graph is empty.")

    if device_mode in ['clean', 'mlu', 'origin']:
      self.device_mode = device_mode
    else:
      raise ValueError("The device_mode should be in ['clean', 'mlu', 'origin']"
	    ", but {}.".format(device_mode))

    self.origin_graph = input_graph_def
    self.origin_nodes_map = self.create_nodes_map(input_graph_def)
    self.nodes_map = self.create_nodes_map(self.input_graph)

  def create_nodes_map(self, graph):
    """Builds a mapping of node names to their defs from the graph."""
    nodes_map = {}
    for node in graph.node:
      if node.name not in nodes_map.keys():
        nodes_map[node.name] = node
      else:
        raise ValueError("Duplicate node names detected.")
    return nodes_map

  def build_fake_int_graph(self, input_graph):
    """Build a fake int graph for get const.
      Return:
        A fake cpu int graph.
    """
    print("Build fake graph start...")
    output_graph_def = graph_pb2.GraphDef()
    delete_names = []
    output_nodes = {}
    int_ops = []

    # convert MLU only ops to corresponding CPU ops
    if 'FC' in self.int_op_list:
      int_ops.extend(['MatMul', 'MLP'])
    if 'BatchMatMul' in self.int_op_list:
      int_ops.extend(['BatchMatMul', 'BatchMatMulV2'])
    for node in input_graph.node:
      if node.op in int_ops:
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        fake_quant_input = create_node(
                            op = self.FAKEQUANTSCALEINTGEN_OP_NAME,
                            inputs = [new_node.input[0]],
                            name = new_node.name + '/fake_quant_int_inputs')
        new_node.input[0] = fake_quant_input.name
        fake_quant_filter = create_node(
                            op = self.FAKEQUANTSCALEINTGEN_OP_NAME,
                            inputs = [new_node.input[1]],
                            name = new_node.name + '/fake_quant_int_weights')
        new_node.input[1] = fake_quant_filter.name
        bias_node = None
        # MLP isn't regitered on cpu, split it into MatMul and BiasAdd.
        if node.op in ['MLP']:
          matmul_node = create_node(
                                    op = 'MatMul',
                                    inputs = [new_node.input[0], new_node.input[1]],
                                    name = new_node.name + '_frommlp')
          matmul_node.attr['T'].CopyFrom(new_node.attr['T'])
          bias_node = create_node(
                               op = 'BiasAdd',
                               inputs = [matmul_node.name, new_node.input[2]],
                               name = new_node.name)
          bias_node.attr['T'].CopyFrom(new_node.attr['T'])
          new_node = matmul_node

        delete_names.extend([node.name])
        if bias_node:
          output_nodes[node.name] = [fake_quant_input, fake_quant_filter,
                                     new_node, bias_node]
        else:
          output_nodes[node.name] = [fake_quant_input, fake_quant_filter, new_node]

    for node in input_graph.node:
      if node.name in output_nodes.keys():
        output_graph_def.node.extend(output_nodes[node.name])
      elif node.name not in delete_names:
        output_graph_def.node.extend([node])
    if self.output_node_names:
      output_graph = graph_util.extract_sub_graph(output_graph_def,
                                           self.output_node_names)
    for node in output_graph_def.node:
      node.device = '/device:CPU:0'
    print("Build fake graph end...")
    return output_graph_def

  def float32_to_int(self, data):
    """Fix float32 weights to int weights and get position and scale."""


    maximum = np.amax(data)
    minimum = np.amin(data)

    fix_max = lambda pos : self.INT_MAX * (2 ** pos)
    fix_min = lambda pos : self.INT_MIN * (2 ** pos)

    for pos in range(-self.POS_RANGE_ABS, self.POS_RANGE_ABS, 1):
      if maximum <= fix_max(pos) and minimum >= fix_min(pos):
        position = pos
        break

    #scale = fix_max(position) / maximum if abs(maximum) >= abs(minimum) \
    #        else abs(fix_min(position) / minimum)

    int_data = np.round(data / (2 ** position)).astype(self.NP_INT_TYPE)
    return int_data, position

  def get_input_max_min(self, input_dicts, batch_size):
    """Get every matmul input's max and min.
    Args:
      input_dicts: A dict of dicts, {input_node_name1 : input_data1,
                                    input_node_name2 : input_data2 ,
                                    ...}.
    """
    if not input_dicts:
      raise ValueError("Input Dict is empty.")
    in_dict = dict()
    int_ops = []
    if 'FC' in self.int_op_list:
      int_ops.extend(['MatMul'])
    if 'BatchMatMul' in self.int_op_list:
      int_ops.extend(['BatchMatMul', 'BatchMatMulV2'])

    iters = len(input_dicts)
    count = 0
    with ops.Graph().as_default():
      importer.import_graph_def(self.input_graph, name="")
      config = config_pb2.ConfigProto(allow_soft_placement=True,
                              inter_op_parallelism_threads=1,
                              intra_op_parallelism_threads=1)
      config.graph_options.rewrite_options.mlu_optimizer = 2
      with session.Session(config=config) as sess:
        init = variables.global_variables_initializer()
        sess.run(init)
        for node in self.input_graph.node:
          if node.op in ["MatMul"] and node.op in int_ops:
            prefix = node.name
            if self.pre_node(node).op not in [self.FAKEQUANTSCALEINTGEN_OP_NAME]:
              raise ValueError("Input {} of node '{}' is missing.".format(self.FAKEQUANTSCALEINTGEN_OP_NAME, node.name))
            if self.pre_node(node, 1).op not in [self.FAKEQUANTSCALEINTGEN_OP_NAME]:
              raise ValueError("Weight {} of node '{}' is missing.".format(self.FAKEQUANTSCALEINTGEN_OP_NAME, node.name))
            in_dict[prefix + '/max'] = self.pre_node(node).name + ":3"
            in_dict[prefix + '/min'] = self.pre_node(node).name + ":4"
          elif node.op in ["BatchMatMul", "BatchMatMulV2"] and node.op in int_ops:
            prefix = node.name
            if self.pre_node(node).op not in [self.FAKEQUANTSCALEINTGEN_OP_NAME]:
              raise ValueError("Input {} of node '{}' is missing.".format(self.FAKEQUANTSCALEINTGEN_OP_NAME, node.name))
            if self.pre_node(node, 1).op not in [self.FAKEQUANTSCALEINTGEN_OP_NAME]:
              raise ValueError("Weight {} of node '{}' is missing.".format(self.FAKEQUANTSCALEINTGEN_OP_NAME, node.name))
            in_dict[prefix + ':0/max'] = self.pre_node(node).name + ":3"
            in_dict[prefix + ':0/min'] = self.pre_node(node).name + ":4"
            in_dict[prefix + ':1/max'] = self.pre_node(node, 1).name + ":3"
            in_dict[prefix + ':1/min'] = self.pre_node(node, 1).name + ":4"
        for input_dict in input_dicts:
          input_feed_dict = {
	  	  sess.graph.get_tensor_by_name(name + ":0") : value \
	  	  for name, value in input_dict.items()
          }
          print("Start to get input max min .......")
          input_max_min = sess.run(in_dict, feed_dict=input_feed_dict)

          for key, value in input_max_min.items():
            prefix, suffix = key.rsplit("/", 1)
            if suffix == "max":
              if prefix in self.input_max.keys():
                old_max = self.input_max[prefix]
                new_max = value
                self.input_max[prefix] = new_max if new_max > old_max else old_max
              else:
                self.input_max[prefix] = value
            else:
              if prefix in self.input_min.keys():
                old_min = self.input_min[prefix]
                new_min = value
                self.input_min[prefix] = new_min if new_min < old_min else old_min
              else:
                self.input_min[prefix] = value
          count = count + batch_size
          print("quantized %d/%d input features..." % (count, iters* batch_size))

  def get_input_pos_scale(self):
    if not self.input_max:
      raise ValueError("The dict input_max is None, please call get_input_max_min() to initialize it.")
    if not self.input_min:
      raise ValueError("The dict input_min is None, please call get_input_max_min() to initialize it.")
    for key in self.input_max.keys():
      maximum = self.input_max[key]
      minimum = self.input_min[key]
      fix_max = lambda pos : self.INT_MAX * (2 ** pos)
      fix_min = lambda pos : self.INT_MIN * (2 ** pos)
      for pos in range(-self.POS_RANGE_ABS, self.POS_RANGE_ABS, 1):
        if maximum <= fix_max(pos) and minimum >= fix_min(pos):
          position = pos
          break

      #scale = fix_max(position) / maximum if abs(maximum) >= abs(minimum) \
      #        else abs(fix_min(position) / minimum)
      #self.const_dict[key + "/in_scale"] = scale

      self.const_dict[key + "/in_position"] = pos

  def get_weight_pos_scale(self):
    """Get all weights positions, scales and int_weights."""
    int_ops = []
    if 'FC' in self.int_op_list:
      int_ops.extend(['MatMul', 'MLP'])
    if 'BatchMatMul' in self.int_op_list:
      int_ops.extend(['BatchMatMul', 'BatchMatMulV2'])

    weight_dict = dict()
    with ops.Graph().as_default():
      importer.import_graph_def(self.input_graph, name="")
      config = config_pb2.ConfigProto(allow_soft_placement=True,
                              inter_op_parallelism_threads=1,
                              intra_op_parallelism_threads=1)
      config.graph_options.rewrite_options.mlu_optimizer = 2
      with session.Session(config=config) as sess:
        init = variables.global_variables_initializer()
        sess.run(init)
        for node in self.input_graph.node:
          if node.op in int_ops:
            if node.op not in ['BatchMatMul', 'BatchMatMulV2']:
              prefix = node.name
              if self.pre_node(node).op not in [self.FAKEQUANTSCALEINTGEN_OP_NAME]:
                raise ValueError("Input {} of node '{}' is missing.".format(self.FAKEQUANTSCALEINTGEN_OP_NAME, node.name))
              if self.pre_node(node, 1).op not in [self.FAKEQUANTSCALEINTGEN_OP_NAME]:
                raise ValueError("Weight {} of node '{}' is missing.".format(self.FAKEQUANTSCALEINTGEN_OP_NAME, node.name))
              weight_dict[prefix + '/w_fp32_data'] = self.pre_node(node, 1).input[0] + ":0" \
                  if ":" not in self.pre_node(node, 1).input[0] else  self.pre_node(node, 1).input[0]
        weight_dict = sess.run(weight_dict)
        for key, value in weight_dict.items():
          prefix = key.rsplit("/", 1)[0]
          int_data, position = self.float32_to_int(value)
          weight_node_name = self.pre_node(self.pre_node(self.nodes_map[prefix], 1)).name
          self.const_dict[weight_node_name + "/int_data"] = int_data
          self.const_dict[prefix + "/w_position"] = position

  def pre_node(self, node, idx=0):
    """get node's input node"""
    if idx > len(node.input):
      raise ValueError("Index should be less than len(node.input), but {} vs. {}".format(idx, len(node.input)))
      return
    return self.nodes_map[self._node_name(node.input[idx])]

  def reshape_attr_kernel(self, value): # input shape (768, 768)
    value = value.transpose(1, 0) # shape (768,  768)
    value = value.reshape(12, 64, 4, 192) # shape (12, 64, 4, 192)
    value = value.transpose(0, 2, 1, 3) # shape (12, 4, 64, 192)
    return value.reshape(768, 768)

  def reshape_key_kernel(self, value): # input shape (768, 768)
    value = value.transpose(1, 0) # shape (768,  768)
    value = value.reshape(12, 64, 8, 96) # shape (12, 64, 8, 96)
    value = value.transpose(0, 2, 1, 3) # shape (12, 8, 64, 96)
    return value.reshape(768, 768)

  def reshape_inter_kernel(self, value): # input shape (768, 3072)
    value = value.transpose(1, 0) # shape (3072, 768)
    value = value.reshape(16, 3, 64, 768) # shape (16, 3, 64, 768)
    value = value.transpose(0, 2, 1, 3) # shape (16, 64, 3, 768)
    value = value.reshape(16, 64, 16, 144) # shape (16, 64, 16, 144)
    value = value.transpose(0, 2, 1, 3) # shape (16, 16, 64, 144)
    return value.reshape(3072, 768)

  def reshape_output_kernel(self, value): # input shape (3072, 768)
    value = value.transpose(1, 0) # shape (768, 3072)
    value = value.reshape(4, 3, 64, 4, 768) # shape (4, 3, 64, 4, 768)
    value = value.transpose(0, 3, 2, 1, 4) # shape (4, 4, 64, 3, 768)
    value = value.reshape(16, 64, 8, 288) # shape (16, 64, 8, 288)
    value = value.transpose(0, 2, 1, 3) # shape (16, 8, 64, 288)
    return value.reshape(768, 3072)

  def set_device(self, output_graph):
    print("device_mode: ", self.device_mode)
    if self.device_mode == 'mlu':
      device = '/device:MLU:0'
    elif self.device_mode == 'clean':
      device = ''
    else:
      return output_graph
    for node in output_graph.node:
      node.device = device
    return output_graph

  def _node_name(self, n):
    if n.startswith("^"):
      return n[1:]
    else:
      return n.split(":")[0]

  def rewrite_int_graph(self):
    """Rewrite int graph of Bert Op"""
    print("Rewrite int graph start...")
    self.get_weight_pos_scale()
    self.get_input_pos_scale()
    output_nodes = []
    layers_name = ['bert/encoder/layer_%d/' % i for i in range(self.layer_num)]
    scope_names = self.scope_names
    post_process_name = self.post_process_name

    encoder_filter_idx = [10, 12, 14, 16, 20, 22]
    post_processer_filter_idx = [8]

    for node in self.origin_graph.node:
      if node.op == "BertSquad":
        bert_node = node
        break

    # build int graph
    set_attr_dtype(bert_node, "FilterT", dtypes.int16)
    set_attr_bool(bert_node, "transpose", True)
    # every encoder filter is a stack of filters of 12 layers
    for i in encoder_filter_idx:
      stack_node =  self.origin_nodes_map[self._node_name(bert_node.input[i])]
      set_attr_dtype(stack_node, "T", dtypes.int16)
      for j in range(len(stack_node.input)):
        filter_node_name = stack_node.input[j]
        value = self.const_dict[filter_node_name + "/int_data"]
        if i in [10, 14, 16]:
          value = self.reshape_attr_kernel(value)
        elif i == 12:
          value = self.reshape_key_kernel(value)
        elif i == 20:
          value = self.reshape_inter_kernel(value)
        elif i == 22:
          value = self.reshape_output_kernel(value)
        else:
          raise ValueError("Index error")
        int_filter_node = create_constant_node(filter_node_name + "/int_data", value, self.DTYPE_INT_TYPE, value.shape)
        self.origin_graph.node.extend([int_filter_node])
        stack_node.input[j] = int_filter_node.name + ':0'

    for i in post_processer_filter_idx:
      filter_node_name = self._node_name(bert_node.input[i])
      value = self.const_dict[filter_node_name + "/int_data"]
      int_filter_node = create_constant_node(filter_node_name + "/int_data", value, self.DTYPE_INT_TYPE, value.shape)
      self.origin_graph.node.extend([int_filter_node])
      bert_node.input[i] = int_filter_node.name + ':0'

    # prepare attribute encoder_position and post_processer_position
    positions = []
    for layer_name in layers_name:
      position = []
      for scope_name in scope_names:
        node_name = layer_name + scope_name
        if self.nodes_map[node_name].op in ['MatMul']:
          position.append(self.const_dict[node_name + '/in_position'])
          position.append(self.const_dict[node_name + '/w_position'])
        if self.nodes_map[node_name].op in ['BatchMatMul', 'BatchMatMulV2']:
          position.append(self.const_dict[node_name + ':0/in_position'])
          position.append(self.const_dict[node_name + ':1/in_position'])
      positions.append(position)
    encoder_position = np.array(positions)
    set_attr_int_list(bert_node, 'encoder_position', encoder_position.reshape(2 * len(layers_name) * len(scope_names)))

    post_processer_position = []
    for node_name in post_process_name:
      post_processer_position.append(self.const_dict[node_name + '/in_position'])
      post_processer_position.append(self.const_dict[node_name + '/w_position'])
    set_attr_int_list(bert_node, 'post_processer_position', post_processer_position)

    if self.quantization_output_node_names:
      output_graph = graph_util.extract_sub_graph(self.origin_graph,
          self.quantization_output_node_names)
    else:
      raise Exception("quantization_output_node_names are required, can't be None")
    output_graph = self.set_device(output_graph)
    print("Rewrite int graph end...")
    return output_graph
