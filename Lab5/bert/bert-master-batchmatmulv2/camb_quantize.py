from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

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

def set_attr_float_list(node, key, value):
  list_value = attr_value_pb2.AttrValue.ListValue(f=value)
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


def compute_quantization_position_and_scale(bit_width, data_max, data_min):
  fix_max = lambda pos : (2 ** (bit_width - 1) - 1) * (2 ** pos)
  fix_min = lambda pos : -(2 ** (bit_width - 1)) * (2 ** pos)
  for pos in range(-2 * bit_width, 2 * bit_width, 1):
    if data_max <= fix_max(pos) and data_min >= fix_min(pos):
      position = pos
      break
  scale = fix_max(position) / data_max if abs(data_max) >= abs(data_min) \
          else abs(fix_min(position) / data_min)
  return position, scale


def fp_to_int(data, bit_width, data_max, data_min):
  """Quantize float data to fix integer"""
  int_type_dict = {8:np.int8, 16:np.int16}
  position, scale = compute_quantization_position_and_scale(bit_width, data_max, data_min)
  int_data = np.round(data * scale * (2 ** -position)).astype(int_type_dict[bit_width])
  return int_data, scale, position


def int_to_fp(int_data, scale, position):
  return int_data.astype(np.float16) * (2 ** position) / scale

def threshold_search(bit_width, data, search_grid = 32, symmetric = True):
  data_max = np.amax(data)
  data_min = np.amin(data)
  data_abs_max = max(data_max, np.abs(data_min))
  data_abs_mean = np.mean(np.abs(data))
  distance = float("inf")

  # First search to locate the max/min threshold approximately
  for i in range(0, search_grid):
    if symmetric:
      threshold_max_i = data_abs_max / search_grid * (i + 1)
      threshold_min_i = -threshold_max_i
    else:
      step = (data_max - data_min) / search_grid
      threshold_max_i = data_max - step * i
      threshold_min_i = data_min

    # Try the max/min threshold to find one which can reach minimum differece with original data abs mean
    int_data, scale, position = fp_to_int(data, bit_width, threshold_max_i, threshold_min_i)
    float_data = int_to_fp(int_data, scale, position)

    new_distance = abs(np.mean(np.abs(float_data)) - data_abs_mean) / data_abs_mean
    if new_distance < distance:
      distance = new_distance
      threshold_max = threshold_max_i
      threshold_min = threshold_min_i
      best_i = i

  assert(best_i >= 0 and best_i < search_grid)

  # Second search to find any better max/min threshold around the one found in first search
  for i in range(0, search_grid):
    if symmetric:
      step = data_abs_max * 2 / (search_grid ** 2)
      threshold_max_i = data_abs_max / search_grid * best_i + step * i
      threshold_min_i = -threshold_max_i
    else:
      step = (data_max - data_min) / search_grid
      step2 = step * 2 / search_grid
      threshold_max_i = data_max - step * (best_i + 1) + step2 * i
      threshold_min_i = data_min

    int_data, scale, position = fp_to_int(data, bit_width, threshold_max_i, threshold_min_i)
    float_data = int_to_fp(int_data, scale, position)

    new_distance = abs(np.mean(np.abs(float_data)) - data_abs_mean) / data_abs_mean
    if new_distance > distance:
      distance = new_distance
      threshold_max = threshold_max_i
      threshold_min = threshold_min_i

  return threshold_max, threshold_min


def fp_to_int_naive(bit_width, data):
  '''Just use data max and min for quantization'''
  return fp_to_int(data, bit_width, np.amax(data), np.amin(data))


def fp_to_int_threshold_search(bit_width, data):
  threshold_max, threshold_min = threshold_search(bit_width, data)
  print("Data max:{}, min:{}".format(np.amax(data), np.amin(data)))
  print("Threshold max:{}, min:{}".format(threshold_max, threshold_min))
  return fp_to_int(data, bit_width, threshold_max, threshold_min)


class QuantizeGraph(object):

  QUANTIZATION_TYPE_INT8  = "INT8"
  QUANTIZATION_TYPE_INT16 = "INT16"

  FP_TO_INT = {
            "naive":fp_to_int_naive,
            "threshold_search":fp_to_int_threshold_search
            }

  def __init__(self,
               input_graph_def,
               output_tensor_names=[],
               use_convfirst=False,
               quantization_type=QUANTIZATION_TYPE_INT8,
               channel_quantization=False,
               weight_quantization_alg="naive",
               activation_quantization_alg="naive",
               convfirst_params={
                   'color_mode' : 'rgb',
                   'mean_r' : 0.0,
                   'mean_g' : 0.0,
                   'mean_b' : 0.0,
                   'input_std' : 1.0},
               device_mode='clean',
               int_op_list=['Conv', 'FC', 'LRN'],
               debug = False,
               model_name=''):
    """Sets up the class to fix input graph.

    Args:
      input_graph_def: a instance of tf.GraphDef,
        the graph that will be converted to int graph.
      output_tensor_names: A list, contain all names of output tensor.
      use_convfirts: Bool, whether or not to use convfirst. if use,
	    please specify convfirst_params.
      quantize_width: A string, must be any of [QUANTIZATION_TYPE_INT8, QUANTIZATION_TYPE_INT16]
      convfirst_params: A dict, specify the color_mode, mean_r, mean_g,
	    mean_b and input_std.
      device_node: A string, must be in ['clean', 'mlu',  'origin'].
        clean - delete all operation's devece.
        mlu - set all operation on mlu device.
        origin - don't change any operation's device.
      int_op_list: A list of string, to choose which layer to int.
    """
    self.input_max = dict()
    self.input_min = dict()
    self.input_data = dict()
    self.const_dict = dict()
    self.unable_fold_bias_nodes = []
    self.output_tensor_names = output_tensor_names
    self.output_node_names = [node_name.split(':')[0] for node_name in output_tensor_names]
    self.int_op_list = int_op_list
    self.channel_quantization = channel_quantization
    self.search_grid = 32
    self.debug = debug
    self.model_name_ = model_name

    if weight_quantization_alg not in QuantizeGraph.FP_TO_INT.keys():
        raise ValueError("Quantization alg {} is UNKNOWN.".format(weight_quantization_alg))
    self.weight_fp_to_int = QuantizeGraph.FP_TO_INT[weight_quantization_alg]

    if activation_quantization_alg not in QuantizeGraph.FP_TO_INT.keys():
        raise ValueError("Quantization alg {} is UNKNOWN.".format(activation_quantization_alg))
    self.input_fp_to_int = QuantizeGraph.FP_TO_INT[activation_quantization_alg]

    if quantization_type == QuantizeGraph.QUANTIZATION_TYPE_INT8:
        self.POS_RANGE_ABS = 16
        self.NP_INT_TYPE = np.int8
        self.BIT_WIDTH = 8
        self.DTYPE_INT_TYPE = dtypes.int8
        self.INTMATMUL_OP_NAME = "Int8MatMul"
        self.INTMLP_OP_NAME    = "Int8MLP"
        self.INTLRN_OP_NAME = "Int8LRN"
        self.INTCONV2D_OP_NAME = "Int8Conv2D"
        self.INTCONV2DBIAS_OP_NAME = "Int8Conv2DBias"
        self.FAKEQUANTSCALEINTGEN_OP_NAME = "FakeQuantScaleInt8Gen"
        self.INTBATCHMATMULV2_OP_NAME = "Int8BatchMatMulV2"
    elif quantization_type == QuantizeGraph.QUANTIZATION_TYPE_INT16:
        self.POS_RANGE_ABS = 32
        self.NP_INT_TYPE = np.int16
        self.BIT_WIDTH = 16
        self.DTYPE_INT_TYPE = dtypes.int16
        self.INTMATMUL_OP_NAME = "Int16MatMul"
        self.INTMLP_OP_NAME    = "Int16MLP"
        self.INTLRN_OP_NAME = "Int16LRN"
        self.INTCONV2D_OP_NAME = "Int16Conv2D"
        self.INTCONV2DBIAS_OP_NAME = "Int16Conv2DBias"
        self.FAKEQUANTSCALEINTGEN_OP_NAME = "FakeQuantScaleInt16Gen"
        self.INTBATCHMATMULV2_OP_NAME = "MLUBatchMatMulV2"
    else:
      raise Exception("Quantization type {} is not supported".format(quantization_type))

    if input_graph_def:
      self.in_while_node_names = self.get_quant_ops_in_while_loop(input_graph_def)
      self.input_graph = self.build_fake_int_graph(input_graph_def)
    else:
      raise ValueError("Input graph is empty.")
    self.use_convfirst = use_convfirst
    if self.use_convfirst:
      if convfirst_params['color_mode'] in ['rgb', 'bgr']:
        self.convfirst_params = convfirst_params
      else:
        raise ValueError("The convfirst parameter color_mode should be"
          " in ['rgb', 'bgr'], but {}.".format(convfirst_params['color_node']))
    if device_mode in ['clean', 'mlu', 'origin']:
      self.device_mode = device_mode
    else:
      raise ValueError("The device_mode should be in ['clean', 'mlu', 'origin']"
	    ", but {}.".format(device_mode))
    self.nodes_map = self.create_nodes_map(self.input_graph)
    self.name_to_output_names = self.create_name_to_output_names(self.input_graph)

  def get_quant_ops_in_while_loop(self, graph_def):
    """Find all conv2d/matmul/lrn/BatchMatMulV2 in while loop.
    """
    while_loop_quant_ops = []
    while_loop_scope = []
    for node in graph_def.node:
      if node.op == "LoopCond" and "LoopCond" in node.name:
        while_loop_name = node.name.rsplit("/", 1)[0]
        while_loop_scope.append(while_loop_name)
    for node in graph_def.node:
      if node.op in ["Conv2D", "MatMul", "LRN", "BatchMatMulV2"]:
        for scope in while_loop_scope:
          if scope in node.name:
            while_loop_quant_ops.append(node.name)
            break
    return while_loop_quant_ops

  def create_histogram_for_debug(self, data, title):
    if self.debug:
      histogram = plt.hist(data.flatten(), bins="auto")
      plt.title(title)
      plt.savefig(title + ".png")


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
      input: Must be a cpu graph.
      Return:
        A fake int graph.
    """
    output_graph_def = graph_pb2.GraphDef()
    delete_names = []
    output_nodes = {}
    int_ops = []
    if 'Conv' in self.int_op_list:
      int_ops.extend(['Conv2D', 'FusedConv2DBias'])
    if 'FC' in self.int_op_list:
      int_ops.extend(['MatMul', 'MLP'])
    if 'BatchMatMul' in self.int_op_list:
      int_ops.extend(['BatchMatMulV2'])
    for node in input_graph.node:
      if node.name in self.in_while_node_names:
        continue
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
        if node.op in ['FusedConv2DBias']:
          conv_node = create_node(
                               op = 'Conv2D',
                               inputs = [new_node.input[0], new_node.input[1]],
                               name = new_node.name + '_fromfusedconv2dbias')
          conv_node.attr['T'].CopyFrom(new_node.attr['T'])
          conv_node.attr['strides'].CopyFrom(new_node.attr['strides'])
          conv_node.attr['padding'].CopyFrom(new_node.attr['padding'])
          conv_node.attr['data_format'].CopyFrom(new_node.attr['data_format'])
          bias_node = create_node(
                               op = 'BiasAdd',
                               inputs = [conv_node.name, new_node.input[2]],
                               name = new_node.name)
          bias_node.attr['T'].CopyFrom(new_node.attr['T'])
          new_node = conv_node
        elif node.op in ['MLP']:
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
    if input_graph.library:
      output_graph_def.library.CopyFrom(input_graph.library)
    return output_graph_def


  def get_input_max_min(self, input_dicts, batch_size):
    """Get every conv2d|matmul|lrn|BatchMatMulV2 input's max and min.
    Args:
      num_runs: A integer, run number.
      feed_dict_fn: A function, it return a feed_dict.
           {input_tensor_name1 : input_data1, input_tensor_name2 : input_data2}
    """
    in_dict = dict()
    lrn_dict = dict()
    int_ops = []
    if 'Conv' in self.int_op_list:
      int_ops.extend(['Conv2D'])
    if 'FC' in self.int_op_list:
      int_ops.extend(['MatMul'])
    if 'LRN' in self.int_op_list:
      int_ops.extend(['LRN'])
    if 'BatchMatMul' in self.int_op_list:
      int_ops.extend(['BatchMatMulV2'])

    with ops.Graph().as_default():
      importer.import_graph_def(self.input_graph, name="")
      config = config_pb2.ConfigProto(allow_soft_placement=True,
                              inter_op_parallelism_threads=1,
                              intra_op_parallelism_threads=1)
      config.mlu_options.visible_device_list = "-1"
      config.graph_options.rewrite_options.mlu_optimizer = 2
      with session.Session(config=config) as sess:
        init = variables.global_variables_initializer()
        sess.run(init)
        for node in self.input_graph.node:
          if node.name in self.in_while_node_names:
            continue
          if node.op in ["Conv2D", "MatMul"] and node.op in int_ops:
            prefix = node.name
            if self.pre_node(node).op not in [self.FAKEQUANTSCALEINTGEN_OP_NAME]:
              raise ValueError("Input {} of node '{}' is missing.".format(self.FAKEQUANTSCALEINTGEN_OP_NAME, node.name))
            if self.pre_node(node, 1).op not in [self.FAKEQUANTSCALEINTGEN_OP_NAME]:
              raise ValueError("Weight {} of node '{}' is missing.".format(self.FAKEQUANTSCALEINTGEN_OP_NAME, node.name))
            in_dict[prefix + '/max'] = self.pre_node(node).name + ":3"
            in_dict[prefix + '/min'] = self.pre_node(node).name + ":4"
            in_dict[prefix + '/data'] = self.pre_node(node).name + ":0"
          elif node.op in ['LRN'] and node.op in int_ops:
            in_dict[node.name + "/lrn_input_data"] = node.input[0] if ":" in node.input[0] else node.input[0] + ":0"
            lrn_dict[node.name + '/alpha'] = node.attr["alpha"].f
            lrn_dict[node.name + '/depth_radius'] = node.attr["depth_radius"].i
          elif node.op in ["BatchMatMulV2"] and node.op in int_ops:
            prefix = node.name
            if self.pre_node(node).op not in [self.FAKEQUANTSCALEINTGEN_OP_NAME]:
              raise ValueError("Input {} of node '{}' is missing.".format(self.FAKEQUANTSCALEINTGEN_OP_NAME, node.name))
            if self.pre_node(node, 1).op not in [self.FAKEQUANTSCALEINTGEN_OP_NAME]:
              raise ValueError("Weight {} of node '{}' is missing.".format(self.FAKEQUANTSCALEINTGEN_OP_NAME, node.name))
            in_dict[prefix + '/max'] = self.pre_node(node).name + ":3"
            in_dict[prefix + '/min'] = self.pre_node(node).name + ":4"
            in_dict[prefix + '/data'] = self.pre_node(node).name + ":0"
            in_dict[prefix + '/batch_matmul_b/max'] = self.pre_node(node, 1).name + ":3"
            in_dict[prefix + '/batch_matmul_b/min'] = self.pre_node(node, 1).name + ":4"
            in_dict[prefix + '/batch_matmul_b/data'] = self.pre_node(node, 1).name + ":0"
        print("Start to get input max min .......")

        for input_dict in input_dicts:
          input_feed_dict = {
          	  sess.graph.get_tensor_by_name(name + ":0") : value \
          	  for name, value in input_dict.items()
          }
          input_max_min = sess.run(in_dict, feed_dict=input_feed_dict)

          for key, value in input_max_min.items():
            prefix, suffix = key.rsplit("/", 1)
            if suffix == "lrn_input_data":
              alpha = lrn_dict[prefix + "/alpha"]
              scale = lrn_dict[prefix + "/depth_radius"] * 2 + 1
              alpha_inputs = scale * alpha * np.power(value, 2)
              if prefix in self.input_max.keys():
                old_max = self.input_max[prefix]
                new_max = np.amax(alpha_inputs)
                self.input_max[prefix] = new_max if new_max > old_max else old_max
              else:
                self.input_max[prefix] = np.amax(alpha_inputs)
              if prefix in self.input_min.keys():
                old_min = self.input_min[prefix]
                new_min = np.amin(alpha_inputs)
                self.input_min[prefix] = new_min if new_min < old_min else old_min
              else:
                self.input_min[prefix] = np.amin(alpha_inputs)
              if prefix in self.input_data.keys():
                self.input_data[prefix] = np.append(self.input_data[prefix], alpha_inputs)
              else:
                self.input_data[prefix] = alpha_inputs
            else:
              if suffix == "max":
                if prefix in self.input_max.keys():
                  old_max = self.input_max[prefix]
                  new_max = value
                  self.input_max[prefix] = new_max if new_max > old_max else old_max
                else:
                  self.input_max[prefix] = value
              elif suffix == "min":
                if prefix in self.input_min.keys():
                  old_min = self.input_min[prefix]
                  new_min = value
                  self.input_min[prefix] = new_min if new_min < old_min else old_min
                else:
                  self.input_min[prefix] = value
              else:
                if prefix in self.input_data.keys():
                  self.input_data[prefix] = np.append(self.input_data[prefix], value)
                else:
                  self.input_data[prefix] = value

  def get_input_pos_scale(self):
    if not self.input_max:
      raise ValueError("The dict input_max is None, please call get_input_max_min() to initialize it.")
    if not self.input_min:
      raise ValueError("The dict input_min is None, please call get_input_max_min() to initialize it.")
    for key in self.input_data.keys():
      data = self.input_data[key]
      self.create_histogram_for_debug(data, "activation_" + key.replace("/", "_"))
      int_data, scale, pos = self.input_fp_to_int(self.BIT_WIDTH, data)
      if "batch_matmul_b" not in key:
        self.const_dict[key + "/in_scale"] = scale
        self.const_dict[key + "/in_position"] = pos
      else:
        true_key, _ = key.rsplit("/", 1)
        self.const_dict[true_key + "/w_scale"] = scale
        self.const_dict[true_key + "/w_position"] = pos


  def get_weight_pos_scale(self):
    """Get all weights positions, scales and int_weights."""
    int_ops = []
    if 'Conv' in self.int_op_list:
      int_ops.extend(['Conv2D'])
    if 'FC' in self.int_op_list:
      int_ops.extend(['MatMul'])
    weight_dict = dict()
    with ops.Graph().as_default():
      importer.import_graph_def(self.input_graph, name="")
      config = config_pb2.ConfigProto(allow_soft_placement=True,
                              inter_op_parallelism_threads=1,
                              intra_op_parallelism_threads=1)
      config.mlu_options.visible_device_list = "-1"
      config.graph_options.rewrite_options.mlu_optimizer = 2
      with session.Session(config=config) as sess:
        init = variables.global_variables_initializer()
        sess.run(init)
        for node in self.input_graph.node:
          if node.name in self.in_while_node_names:
            continue
          if node.op in int_ops:
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
          self.create_histogram_for_debug(value, "weight_" + key.replace("/", "_"))
          channels = value.shape[-1]
          if self.channel_quantization and channels > 1:
            int_datas = []
            scales = []
            positions = []
            for c in range(channels):
              cdata, cscale, cpos = self.weight_fp_to_int(self.BIT_WIDTH, value[..., c])
              int_datas.append(cdata)
              scales = np.append(scales, cscale)
              positions = np.append(positions, cpos)
            self.const_dict[prefix + "/w_int_data"] = np.stack(int_datas, axis = -1)
            self.const_dict[prefix + "/w_scale"] = scales
            self.const_dict[prefix + "/w_position"] = positions
          else:
            int_data, scale, position = self.weight_fp_to_int(self.BIT_WIDTH, value)
            self.const_dict[prefix + "/w_int_data"] = int_data
            self.const_dict[prefix + "/w_scale"] = scale
            self.const_dict[prefix + "/w_position"] = position

  def write_const(self):
    import json
    int_ops = []
    if 'Conv' in self.int_op_list:
      int_ops.extend(['Conv2D'])
    if 'FC' in self.int_op_list:
      int_ops.extend(['MatMul'])
    pos_list = []
    for node in self.input_graph.node:
      if node.name in self.in_while_node_names:
        continue
      if node.op in int_ops:
        inpoint = self.const_dict[node.name + '/in_position']
        wpoint = self.const_dict[node.name + '/w_position']
        #wint_data = self.const_dict[node.name + '/w_int_data']
      pos_list.append({node.name : {'in_pos' :  inpoint, 'w_pos' : wpoint}})
    with open('./net_positions.json') as f:
      json.dump(pos_list, f, indent = 4)

  def pre_node(self, node, idx=0):
    """get node's input node"""
    if idx > len(node.input):
      raise ValueError("Index should be less than len(node.input), but {} vs. {}".format(idx, len(node.input)))
      return
    return self.nodes_map[self._node_name(node.input[idx])]

  def set_device(self, output_graph):
    if self.device_mode == 'mlu':
      device = '/device:MLU:0'
    elif self.device_mode == 'clean':
      device = ''
    else:
      return output_graph
    for node in output_graph.node:
      if node.name in self.in_while_node_names:
        node.device = '/device:CPU:0'
      else:
        node.device = device
    return output_graph

  def get_node_input_nodes(self, node):
    return [self.pre_node(node, i) for i in range(len(node.input))]

  def check_fold_bias(self, node_list):
    for node in node_list:
      if node.name in self.unable_fold_bias_nodes:
        return False
      elif len(node.input) == 0:
        if node.op not in ["Const"]:
          return False
        else:
          return True
      else:
        new_node_list = self.get_node_input_nodes(node)
        status = self.check_fold_bias(new_node_list)
        if not status:
          return False
    return True

  def _node_name(self, n):
    if n.startswith("^"):
      return n[1:]
    else:
      return n.split(":")[0]

  def create_name_to_output_names(self, graph_def):
    """{name : set(output_node_name, ...)}"""
    name_to_output_names = {}
    for node in graph_def.node:
      node_name = self._node_name(node.name)
      for input_name in node.input:
        input_name = self._node_name(input_name)
        if input_name not in name_to_output_names:
          output_names = set()
          output_names.add(node_name)
          name_to_output_names[input_name] = output_names
        else:
          name_to_output_names[input_name].add(node_name)
    return name_to_output_names

  def enable_fold_to_conv2d_bias(self, node):
    if (node.op in ['BiasAdd', 'Add'] and
        self.pre_node(node).op in ['Conv2D', 'MatMul'] and
        len(self.name_to_output_names[self.pre_node(node).name]) == 1):
        return True
    return False

  def rewrite_int_graph(self):
    """Rewrite int graph."""
    self.get_weight_pos_scale()
    self.get_input_pos_scale()
    delete_nodes_name = []
    output_nodes = {}
    new_add_nodes = {}
    int_ops = []
    if 'Conv' in self.int_op_list:
      int_ops.extend(['Conv2D'])
    if 'FC' in self.int_op_list:
      int_ops.extend(['MatMul'])
    if 'LRN' in self.int_op_list:
      int_ops.extend(['LRN'])
    if 'BatchMatMul' in self.int_op_list:
      int_ops.extend(['BatchMatMulV2'])
    for node in self.input_graph.node:
      if (self.enable_fold_to_conv2d_bias(node) and
        self.pre_node(node).op in int_ops):
        if self.pre_node(node).name in self.in_while_node_names:
          continue
        enable_fold_bias = True
        enable_fold_bias = self.check_fold_bias([self.pre_node(node, 1)])
        if not enable_fold_bias:
          self.unable_fold_bias_nodes.append(node.name)
        else:
          if self.pre_node(node).op in ['Conv2D']:
            int_node = self.fix_conv2d(node.name)
            weights_nodes = self.fix_weights(self.pre_node(node))
            biases_nodes = self.maybe_change_biases(node)
            if "const_replace" in weights_nodes[0].name:
              int_node.input[1] = weights_nodes[0].name
            if "const_replace" in biases_nodes[0].name:
              int_node.input[2] = biases_nodes[0].name
          else:
            int_node = self.fix_matmul(node.name)
            weights_nodes =self.fix_weights(self.pre_node(node))
            biases_nodes = self.maybe_change_biases(node)
            if "const_replace" in weights_nodes[0].name:
              int_node.input[1] = weights_nodes[0].name
            if "const_replace" in biases_nodes[0].name:
              int_node.input[2] = biases_nodes[0].name
          #delete conv2d
          delete_nodes_name.extend([self.pre_node(node).name])
          #delete biasadd or add
          delete_nodes_name.extend([node.name])
          #delete fp32 weights and fp32 identity
          delete_nodes_name.extend([node.name for node in weights_nodes])
          #delete origin biases_sub_graph
          delete_nodes_name.extend([node.name for node in biases_nodes])
          #add int_conv2d_bias
          output_nodes.update({int_node.name : int_node})
          #add int weights and int identity
          output_nodes.update({node.name : node for node in weights_nodes})
          #add folded biases nodes
          output_nodes.update({node.name : node for node in biases_nodes})
    if('efficientNet' in self.model_name_):
      self.fix_swish(delete_nodes_name, new_add_nodes, output_nodes)
    for node in self.input_graph.node:
      if node.name in self.in_while_node_names:
        continue
      if (node.name not in delete_nodes_name and
          node.op in ['Conv2D', 'MatMul', 'BatchMatMulV2']) and node.op in int_ops:
        if node.op in ['Conv2D']:
          int_node = self.fix_conv2d(node.name)
          weights_nodes = self.fix_weights(node)
          if "const_replace" in weights_nodes[0].name:
            int_node.input[1] = weights_nodes[0].name
        elif node.op in ['MatMul']:
          int_node = self.fix_matmul(node.name)
          weights_nodes = self.fix_weights(node)
          if "const_replace" in weights_nodes[0].name:
            int_node.input[1] = weights_nodes[0].name
        else:
          int_node = self.fix_batchmatmulv2(node.name)
          weights_nodes = []
          #weights_nodes = self.fix_weights(node)
          #if "const_replace" in weights_nodes[0].name:
          #  int_node.input[1] = weights_nodes[0].name
        delete_nodes_name.extend([node.name])
        delete_nodes_name.extend([node.name for node in weights_nodes])
        output_nodes.update({int_node.name : int_node})
        output_nodes.update({node.name : node for node in weights_nodes})
      elif (node.name not in delete_nodes_name and node.op in ['LRN']
           and node.op in int_ops):
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        new_node.op = self.INTLRN_OP_NAME
        set_attr_int(new_node, 'position', self.const_dict[node.name + "/in_position"])
        delete_nodes_name.extend([node.name])
        output_nodes.update({node.name : new_node})
    output_graph = graph_pb2.GraphDef()
    # for some folded weights and biases
    for name, node in output_nodes.items():
      if "const_replace" in name:
        output_graph.node.extend([node])

    for node in self.input_graph.node:
      if node.op in [self.FAKEQUANTSCALEINTGEN_OP_NAME]:
        continue
      elif node.name not in delete_nodes_name:
        output_graph.node.extend([node])
      elif node.name in delete_nodes_name and node.name in output_nodes.keys():
        output_graph.node.extend([output_nodes[node.name]])

    for add_node in new_add_nodes:
      output_graph.node.extend([new_add_nodes[add_node]])

    if self.output_node_names:
      output_graph = graph_util.extract_sub_graph(output_graph,
                                           self.output_node_names)
    output_graph = self.set_device(output_graph)
    if self.input_graph.library:
      output_graph.library.CopyFrom(self.input_graph.library)
    return output_graph

  def fix_weights(self, node):
    """Transform weights node from float32 to int.

      Args:
        node: Conv2D or MatMul node

      Return:
        A list of node
    """
    output_nodes = []
    value = self.const_dict[node.name + '/w_int_data']
    fix_w_node = self.pre_node(self.pre_node(node, 1))
    if fix_w_node.op in ['Const']:
      name = fix_w_node.name
      output_nodes.append(create_constant_node(name,
                                value, self.DTYPE_INT_TYPE, value.shape))
    elif (fix_w_node.op in ['Identity'] and
          self.pre_node(fix_w_node).op in ['Const']):
      name = self.pre_node(fix_w_node).name
      iden_node = node_def_pb2.NodeDef()
      iden_node.CopyFrom(fix_w_node)
      set_attr_dtype(iden_node, "T", self.DTYPE_INT_TYPE)
      output_nodes.append(iden_node)
      output_nodes.append(create_constant_node(name,
                                value, self.DTYPE_INT_TYPE, value.shape))
    else :
      name = self.pre_node(node, 1).input[0].replace(":","_") + "/const_replace"
      output_nodes.append(create_constant_node(name,
                                value, self.DTYPE_INT_TYPE, value.shape))
      print("Warning: Does not find weights for node {} in input graph. "
            "Will make the second input of {} to Const.".format(
            node.name, node.name))
    return output_nodes

  def maybe_change_biases(self, node):
    """
      Args:
        node: BiasAdd or Add node

      Return:
        A list of node
    """
    output_nodes = []
    if self.pre_node(node, 1).op in ['Const']:
      output_nodes.extend([self.pre_node(node, 1)])
    elif (self.pre_node(node, 1).op in ['Identity'] and
          self.pre_node(self.pre_node(node, 1)).op in ['Const']):
      output_nodes.extend([self.pre_node(node, 1), self.pre_node(self.pre_node(node, 1))])
    else :
      name = node.input[1].replace(":","_") + "/const_replace"
      biases = node.input[1] if ":" in node.input[1] else node.input[1] + ":0"
      graph = ops.Graph()
      with graph.as_default():
        importer.import_graph_def(self.input_graph)
        config = config_pb2.ConfigProto(allow_soft_placement=True,
                                inter_op_parallelism_threads=1,
                                intra_op_parallelism_threads=1)
        config.mlu_options.visible_device_list = "-1"
        config.graph_options.rewrite_options.mlu_optimizer = 2
        with session.Session(graph=graph, config=config) as sess:
          value = sess.run("import/" + biases)
      output_nodes.extend([create_constant_node(name,
                                value, dtypes.float32, value.shape)])
      print("Warning: Does not find biases for node {} in input graph. "
            "Will make the second input of {} to Const.".format(
            node.name, node.name))
    return output_nodes

  def run_const(self, node):
    """Some `Const` node have inputs control dependence, this method
    will delete the input control dependence and get it's value.

    Args:
	  node: `Const` node.
    Return:
        return the `Const` node value
	"""
    new_node = node_def_pb2.NodeDef()
    new_node.name = node.name
    new_node.op = node.op
    new_node.attr["value"].CopyFrom(node.attr["value"])
    new_node.attr["dtype"].CopyFrom(node.attr["dtype"])
    if node.input:
      new_node.input.extend(node.input)
      for idx, inp in enumerate(new_node.input):
        if inp[0] == '^':
          del new_node.input[idx]
    sub_graph_def = graph_pb2.GraphDef()
    sub_graph_def.node.extend([new_node])
    output_node_name = "import/" + new_node.name + ":0"
    graph = ops.Graph()
    with graph.as_default():
      importer.import_graph_def(sub_graph_def)
    config = config_pb2.ConfigProto(allow_soft_placement=True,
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    config.mlu_options.visible_device_list = "-1"
    config.graph_options.rewrite_options.mlu_optimizer = 2
    with session.Session(graph=graph, config=config) as sess:
      value = sess.run(output_node_name)
    return value

  def sigmoid_replace_swish(self, node_name):
    """if is efficientNet network, create Sigmoid and mul op"""
    node = self.nodes_map[node_name]
    bias = None
    bias = node.input[0]
    sigmoid_inputs = []
    mul_inputs = []
    swish_inputs = []
    if bias:
      sigmoid_inputs.append(bias)
      mul_inputs.append(bias)
    else:
      raise ValueError("node operation should be in [swish],"
              "but {}.".format(node.op))
    op = 'Sigmoid'
    sigmoid_node_name = node_name + '/' + op
    sigmoid_node = create_node(op = op,
                               name = sigmoid_node_name,
                               inputs = sigmoid_inputs)
    set_attr_dtype(sigmoid_node, 'T', dtypes.float32)
    mul_inputs.append(sigmoid_node.name)
    op = 'Mul'
    mul_node_name = node_name + '/' + op
    mul_node = create_node(op = op,
                           name = mul_node_name,
                           inputs = mul_inputs)
    set_attr_dtype(mul_node, 'T', dtypes.float32)

    swish_inputs.append(mul_node.name)
    swish_op = create_node(op = 'Identity',
                           name = node_name,
                           inputs = swish_inputs)
    set_attr_dtype(swish_op, 'T', dtypes.float32)
    return sigmoid_node, mul_node, swish_op

  def fix_swish(self, delete_nodes_name, new_add_nodes, output_nodes):
    """ Transform efficientNet network to swish to Sigmoid and mul"""
    for node in self.input_graph.node:
      if(node.op in ['swish_f32']):
        sigmoid_op, mul_op, swish_op  = self.sigmoid_replace_swish(node.name)
        delete_nodes_name.extend([node.name])
        new_add_nodes.update({sigmoid_op.name : sigmoid_op})
        new_add_nodes.update({mul_op.name : mul_op})
        output_nodes.update({swish_op.name : swish_op})

  def fix_conv2d(self, node_name):
    """Transform Conv2D to Int8Conv2D/Int16Conv2D and Conv2D+BiasAdd to Int8Conv2DBias/Int16Conv2DBias.

    Args:
      node_name: Name of Conv2D or BiasAdd or Add.
    Returns:
      A Int8Conv2D/Int8Conv2DBias/Int16Conv2D/Int16Conv2DBias node.
    """
    node = self.nodes_map[node_name]
    ori_node = node
    bias = None
    if node.op in ['BiasAdd', 'Add'] and self.pre_node(node).op in ['Conv2D']:
      bias = node.input[1]
      node = self.pre_node(node)
    elif node.op not in ['Conv2D']:
      raise ValueError("node operation should be in [BiasAdd, Add, MatMul],"
        "but {}.".format(node.op))
    inputs = [self.pre_node(node).input[0], self.pre_node(node, 1).input[0]]
    op_name = self.INTCONV2D_OP_NAME
    if bias:
      inputs.append(bias)
      op_name = self.INTCONV2DBIAS_OP_NAME

    conv_node = create_node(op = op_name,
                            name = node_name,
                            inputs = inputs)
    set_attr_dtype(conv_node, 'FilterT', self.DTYPE_INT_TYPE)
    conv_node.attr['InT'].CopyFrom(node.attr['T'])
    conv_node.attr['OutT'].CopyFrom(node.attr['T'])
    conv_node.attr['strides'].CopyFrom(node.attr['strides'])
    conv_node.attr['data_format'].CopyFrom(node.attr['data_format'])
    conv_node.attr['padding'].CopyFrom(node.attr['padding'])

    in_position = self.const_dict[node.name + '/in_position']
    in_scale = self.const_dict[node.name + '/in_scale']
    in_scale = in_scale / (2 ** in_position)
    w_position = self.const_dict[node.name + '/w_position']
    w_scale = self.const_dict[node.name + '/w_scale']
    w_scale = w_scale / (2 ** w_position)
    set_attr_float(conv_node, 'input_scale', in_scale)
    if np.size(w_scale) == 1:
      set_attr_float(conv_node, 'filter_scale', w_scale)
    else:
      set_attr_float_list(conv_node, 'filter_scales', w_scale)

    set_attr_int_list(conv_node, 'paddings', [0, 0, 0, 0])
    if self.use_convfirst and \
       self.const_dict[node.name + '/w_int_data'].shape[2] == 3:
      input_std = self.convfirst_params['input_std']
      set_attr_float(conv_node, 'mean_r', self.convfirst_params['mean_r'])
      set_attr_float(conv_node, 'mean_g', self.convfirst_params['mean_g'])
      set_attr_float(conv_node, 'mean_b', self.convfirst_params['mean_b'])
      set_attr_float(conv_node, 'std', input_std)
      set_attr_bool(conv_node, 'use_convfirst', True)
      #TODO(xpf) add PadV2 when conv support to specify pad value.
    if self.pre_node(self.pre_node(node)).op in ['Pad']:
      print('Has Pad before ConvFirst !!!')
      pad_node = self.pre_node(self.pre_node(node))
      if len(self.name_to_output_names[pad_node.name]) > 1:
        raise ValueError("more than one node are dependent on node {}, "
                "Please set use_convfirst=False.".format(pad_node.name))
      if self.pre_node(pad_node, 1).op not in ['Const', 'Identity']:
        raise ValueError("`paddings` of `Pad` must be a `Const` or `Identity`,"
         "but {}. Please set use_convfirst=False.".format(self.pre_node(pad_node).op))
      if self.pre_node(pad_node, 1).op in ['Const']:
        paddings_const = self.pre_node(pad_node, 1)
      else :
        paddings_const = self.pre_node(self.pre_node(pad_node, 1))
      pad_v = self.run_const(paddings_const)
      paddings = [pad_v[1][0], pad_v[1][1], pad_v[2][0], pad_v[2][1]]
      set_attr_int_list(conv_node, 'paddings', paddings)
      conv_node.input[0] = pad_node.input[0]
    return conv_node

  def fix_matmul(self, node_name):
    """Transform MatMul+BiasAdd to Int8MLP and matmul to Int8MatMul.
	Args:
      node_name: BiasAdd, MatMul operation's name
    Return:
      fixed fully connected node.
    """
    node = self.nodes_map[node_name]
    bias = None
    if node.op in ['BiasAdd', 'Add'] and self.pre_node(node).op in ['MatMul']:
      bias = node.input[1]
      node = self.pre_node(node)
    elif node.op not in ['MatMul']:
      raise ValueError("node operation should be in [BiasAdd, 'Add', MatMul],"
	    "but {}.".format(node.op))
    inputs = [self.pre_node(node).input[0], self.pre_node(node, 1).input[0]]
    op_name = self.INTMATMUL_OP_NAME
    if bias:
      inputs.append(bias)
      op_name = self.INTMLP_OP_NAME
    mlp_node = create_node(op = op_name,
                           name = node_name,
                           inputs = inputs)
    set_attr_dtype(mlp_node, 'T1', self.DTYPE_INT_TYPE)
    mlp_node.attr['T'].CopyFrom(node.attr['T'])
    in_position = self.const_dict[node.name + '/in_position']
    in_scale = self.const_dict[node.name + '/in_scale']
    in_scale = in_scale / (2 ** in_position)
    w_position = self.const_dict[node.name + '/w_position']
    w_scale = self.const_dict[node.name + '/w_scale']
    w_scale = w_scale / (2 ** w_position)
    set_attr_float(mlp_node, 'input_scale', in_scale)
    set_attr_float(mlp_node, 'filter_scale', w_scale)
    mlp_node.attr["transpose_a"].CopyFrom(node.attr["transpose_a"])
    mlp_node.attr["transpose_b"].CopyFrom(node.attr["transpose_b"])
    return mlp_node

  def fix_batchmatmulv2(self, node_name):
      """Transform batchmatmulv2 to MLUBatchMatMulV2.
  	Args:
        node_name: name of BatchMatMulV2
      Return:
        A BatchMatMulV2 node.
      """
      node = self.nodes_map[node_name]
      inputs = [self.pre_node(node).input[0], self.pre_node(node, 1).input[0]]
      op_name = self.INTBATCHMATMULV2_OP_NAME
      batchmatmulv2_node = create_node(op = op_name,
                                       name = node_name,
                                       inputs = inputs)
      batchmatmulv2_node.attr['T'].CopyFrom(node.attr['T'])
      in_position = self.const_dict[node.name + '/in_position']
      in_scale = self.const_dict[node.name + '/in_scale']
      w_position = self.const_dict[node.name + '/w_position']
      w_scale = self.const_dict[node.name + '/w_scale']
      set_attr_float(batchmatmulv2_node, 'input1_scale', in_scale)
      set_attr_float(batchmatmulv2_node, 'input2_scale', w_scale)
      set_attr_int(batchmatmulv2_node, 'input1_position', in_position)
      set_attr_int(batchmatmulv2_node, 'input2_position', w_position)
      batchmatmulv2_node.attr["adj_x"].CopyFrom(node.attr["adj_x"])
      batchmatmulv2_node.attr["adj_y"].CopyFrom(node.attr["adj_y"])
      return batchmatmulv2_node
