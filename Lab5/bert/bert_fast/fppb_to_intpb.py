import tensorflow as tf
import numpy as np
import argparse
import os
import ConfigParser
import my_camb_quantize
import collections
import tokenization
from run_squad import read_squad_examples, convert_examples_to_features
#import preprocess

def read_pb(input_model_name):
    input_graph_def = tf.GraphDef()
    with tf.gfile.GFile(input_model_name,"rb") as f:
        input_graph_def.ParseFromString(f.read())
        f.close()
    return input_graph_def

def create_pb(output_graph_def, output_model_name):
    with tf.gfile.GFile(output_model_name, "wb") as f:
        f.write(output_graph_def.SerializeToString())
        f.close()
    print("cpu_pb transform to mlu_int_pb finished")

def process_feature_batch(features, batch_size, iter_idx):
  current_features = features[iter_idx * batch_size: (iter_idx + 1) * batch_size]
  if len(current_features) != batch_size:
      raise ValueError("no enough features! please input a smaller iter!")
  input_ids_list = list()
  input_mask_list = list()
  segment_ids_list = list()

  for feature in current_features:
    input_ids_list.append(feature.input_ids)
    input_mask_list.append(feature.input_mask)
    segment_ids_list.append(feature.segment_ids)

  input_ids = np.asarray(input_ids_list)
  input_mask = np.asarray(input_mask_list)
  segment_ids = np.asarray(segment_ids_list)
  return [input_ids, input_mask, segment_ids]

def process_data_and_get_input_max_min(data_list,
                                       fixer,
                                       input_node_names,
                                       iters,
                                       vocab_file,
                                       do_lower_case,
                                       seq_length,
                                       doc_stride=128,
                                       max_query_length=64,
                                       batch_size=8,
                                       preprocess_fn="default_preprocess"):
  """Precess input data and get input max and min.
  """
  eval_features = []

  def append_feature(feature):
    eval_features.append(feature)

  eval_examples = read_squad_examples(input_file=data_list, is_training=False)
  eval_examples = eval_examples[0:batch_size*iters]
  tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

  convert_examples_to_features(examples=eval_examples,
                               tokenizer=tokenizer,
                               max_seq_length=seq_length,
                               doc_stride=doc_stride,
                               max_query_length=max_query_length,
                               is_training=False,
                               output_fn=append_feature)
  input_dicts = []
  for i in range(iters):
    inputs = process_feature_batch(eval_features, batch_size, i)
    input_dict = dict(zip(input_node_names, inputs))
    input_dicts.append(input_dict)
  fixer.get_input_max_min(input_dicts, batch_size)

  print("quantize input end")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('params_file',
                         help = 'Please input params_file(xxx.ini).')

    args = parser.parse_args()
    config = ConfigParser.ConfigParser()
    config.read(args.params_file)

    model_param = dict(config.items('model'))
    data_param = dict(config.items('data'))
    config_param = dict(config.items('config'))

    # model_param
    input_graph_def = read_pb(model_param['original_models_path'])
    layer_num = int(model_param['layer_num'])
    seq_length = int(model_param['seq_length'])
    input_node_names = model_param['input_nodes'].replace(" ","").split(",")
    if len(input_node_names) != 3:
      raise ValueError("The demo expect 3 inputs,"
	                   " but number of input is {}".format(len(input_node_names)))

    output_node_names = model_param['output_nodes'].replace(" ","").split(",")
    quantization_output_node_names = model_param['quantization_output_nodes'].replace(" ","").split(",")
    scope_names = model_param['scope_names'].replace(" ","").replace("\n","").split(",")
    post_process_name = model_param['post_process_name'].replace(" ","").split(",")
    output_model_name = model_param['save_model_path']

    # data_param
    data_list = os.getenv('TENSORFLOW_MODELS_DATA_HOME') + '/' + data_param['data_list']
    vocab_file = os.getenv('TENSORFLOW_MODELS_DATA_HOME') + '/' + data_param['vocab_file']
    do_lower_case = bool(data_param['do_lower_case'])
    doc_stride = int(data_param['doc_stride'])
    max_query_length = int(data_param['max_query_length'])
    batch_size = int(data_param['batch_size'])
    iters = int(data_param['iters'])

    # config_param
    int_op_list = ["FC", "BatchMatMul"]
    if 'int_op_list' in config_param:
      int_op_list = config_param['int_op_list'].replace(" ","").split(",")

    device_mode = 'clean'
    if 'device_mode' in config_param:
      device_mode = config_param['device_mode']

    quantization_type = my_camb_quantize.QuantizeGraph.QUANTIZATION_TYPE_INT16

    print("input_node_names:  {}".format(input_node_names))
    print("output_node_names: {}".format(output_node_names))
    print("quant_output:      {}".format(quantization_output_node_names))
    print("batch_size:        {}".format(batch_size))
    print("iters:             {}".format(iters))
    print("int_op_list:       {}".format(int_op_list))
    print("device_mode:       {}".format(device_mode))
    print("quantization type: {}".format(quantization_type))

    fixer = my_camb_quantize.QuantizeGraph(
                   input_graph_def = input_graph_def,
                   layer_num = layer_num,
                   output_node_names = output_node_names,
                   quantization_output_node_names = quantization_output_node_names,
                   scope_names = scope_names,
                   post_process_name = post_process_name,
                   quantization_type = quantization_type,
                   device_mode = device_mode,
                   int_op_list = int_op_list)

    process_data_and_get_input_max_min(data_list=data_list,
                                         fixer=fixer,
                                         input_node_names=input_node_names,
                                         iters=iters,
                                         vocab_file=vocab_file,
                                         do_lower_case=do_lower_case,
                                         seq_length=seq_length,
                                         doc_stride=doc_stride,
                                         max_query_length=max_query_length,
                                         batch_size=batch_size,
                                         preprocess_fn = None)

    output_graph_def = fixer.rewrite_int_graph()

    if not os.path.exists(os.path.dirname(output_model_name)):
      os.system("mkdir -p {}".format(os.path.dirname(output_model_name)))
    create_pb(output_graph_def, output_model_name)
