import tensorflow as tf
import numpy as np
import argparse
import os
import ConfigParser
import camb_quantize
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
                                       input_tensor_names,
                                       num_runs,
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
  eval_examples = eval_examples[0:batch_size*num_runs]
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
  input_node_names = [node_name.split(':')[0] for node_name in input_tensor_names]
  for i in range(num_runs):
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
    seq_length = int(model_param['seq_length'])
    input_tensor_names = model_param['input_tensor_names'].replace(" ","").split(",")
    if len(input_tensor_names) != 3:
      raise ValueError("The demo expect 3 inputs,"
	                   " but number of input is {}".format(len(input_tensor_names)))

    output_tensor_names = model_param['output_tensor_names'].replace(" ","").split(",")
    output_model_name = model_param['save_model_path']

    # data_param
    data_list = os.getenv('TENSORFLOW_MODELS_DATA_HOME') + '/' + data_param['data_list']
    vocab_file = os.getenv('TENSORFLOW_MODELS_DATA_HOME') + '/' + data_param['vocab_file']
    do_lower_case = bool(data_param['do_lower_case'])
    doc_stride = int(data_param['doc_stride'])
    max_query_length = int(data_param['max_query_length'])
    batch_size = int(data_param['batch_size'])
    num_runs = int(data_param['num_runs'])

    # config_param
    int_op_list = ["FC"]
    if 'int_op_list' in config_param:
      int_op_list = config_param['int_op_list'].replace(" ","").split(",")

    device_mode = 'clean'
    if 'device_mode' in config_param:
      device_mode = config_param['device_mode']

    quantization_type = camb_quantize.QuantizeGraph.QUANTIZATION_TYPE_INT16

    print("input_tensor_names:  {}".format(input_tensor_names))
    print("output_tensor_names: {}".format(output_tensor_names))
    print("batch_size:          {}".format(batch_size))
    print("num_runs:            {}".format(num_runs))
    print("int_op_list:         {}".format(int_op_list))
    print("device_mode:         {}".format(device_mode))
    print("quantization type:   {}".format(quantization_type))

    fixer = camb_quantize.QuantizeGraph(
                   input_graph_def = input_graph_def,
                   output_tensor_names = output_tensor_names,
                   quantization_type = quantization_type,
                   device_mode = device_mode,
                   int_op_list = int_op_list)

    process_data_and_get_input_max_min(
            data_list=data_list,
            fixer=fixer,
            input_tensor_names=input_tensor_names,
            num_runs=num_runs,
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
