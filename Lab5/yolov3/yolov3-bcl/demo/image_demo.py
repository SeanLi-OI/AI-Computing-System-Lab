#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import argparse
import cv2
import os
import numpy as np
import time
import core.utils as utils
import tensorflow as tf
from PIL import Image

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph

def get_sample_counts(filename):
  count = 0
  for record in tf.python_io.tf_record_iterator(path=filename):
    count += 1
  return count


if __name__ == "__main__":
    pb_file         = "./yolov3_coco.pb"
    file_list         = "./file_list"
    number          = 1
    num_classes     = 80
    input_size      = 544
    graph           = tf.Graph()
    data_parallelism = 1
    model_parallelism = 1
    core_num = 1
    core_version    = "MLU100"
    batch_size = 1
    precision = "float32"
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0",
            "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    output_layer    = ["input/input_data:0", "pred_sbbox/concat_2:0",
            "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--file_list", help="file_list to be processed")
    parser.add_argument("--number", type=int, help="number of file_list to be processed")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--data_parallelism", type=int, help="data_parallelism")
    parser.add_argument("--model_parallelism", type=int, help="model_parallelism")
    parser.add_argument("--core_num", type=int, help="core_num")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--precision", help="datatype")
    parser.add_argument("--core_version", help="MLU100")

    args = parser.parse_args()

    if args.file_list:
        file_list = args.file_list
    if args.graph:
        pb_file = args.graph
    if args.output_layer:
        output_layer = args.output_layer
    if args.number:
        number = args.number
    if args.batch_size:
        batch_size = args.batch_size
    if args.data_parallelism:
        data_parallelism = args.data_parallelism
    if args.model_parallelism:
        model_parallelism = args.model_parallelism
    if args.core_num:
        core_num = args.core_num
    if args.precision:
        precision = args.precision
    if args.core_version:
        core_version = args.core_version

  # some check
    if data_parallelism not in [1, 2, 4, 8, 16, 32]:
        print ("Error! data_parallelism should be one of [1, 2, 4, 8, 16, 32]")
        exit(0)
    if model_parallelism not in [1, 2, 4, 8, 16, 32]:
        print ("Error! model_parallelism should be one of [1, 2, 4, 8, 16, 32]")
        exit(0)
    if model_parallelism * data_parallelism > 32:
        print ("Error! model_parallellism * data_parallelism should less than 32.")
        exit(0)
    if data_parallelism > 1:
        if batch_size < data_parallelism:
            print ("Error! batch_size must >= data_parallelism")
            exit(0)
        if batch_size % data_parallelism != 0:
            print ("Error! batch_size must be multiple of data_parallelism")
            exit(0)


    config = tf.ConfigProto(allow_soft_placement=True,
			  inter_op_parallelism_threads=1,
                          intra_op_parallelism_threads=1)
    config.mlu_options.save_offline_model = False
    config.mlu_options.data_parallelism = data_parallelism
    config.mlu_options.model_parallelism = model_parallelism
    config.mlu_options.core_num = core_num
    #config.mlu_options.fusion = True
    config.mlu_options.core_version = core_version
    config.mlu_options.precision = precision

    #config.graph_options.rewrite_options.remapping = 2
    #config.graph_options.rewrite_options.constant_folding = 2
    #config.graph_options.rewrite_options.arithmetic_optimization = 2

    config.mlu_options.optype_black_list ="StridedSlice"

    f = open(file_list)
    file_list_lines = f.readlines()
    sample_counts = len(file_list_lines)
    print("number = ", number,
            "sample_counts = ", sample_counts,
            "batch_size = ", batch_size)
    #if number > sample_counts:
    #  number = sample_counts
    if number < sample_counts:
      number = sample_counts
    if number < batch_size:
      print("Error! number of images must be >= batch_size")
      exit(0)

    graph = load_graph(pb_file)
    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

    all_images = []
    all_images_name = []
    for i in range(number):
        line = file_list_lines[i]
        img_name = line.strip().split("/")[-1]
        original_image = cv2.imread(line.rstrip())
        h,  w, _  = original_image.shape
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        new_h,  new_w, _  = original_image.shape
        original_image_size = original_image.shape[:2]
        all_images_name.append(img_name)
        all_images.append(original_image)

    image_data = utils.images_preporcess(all_images, [input_size, input_size])
    print(image_data.shape)
    run_times = np.ceil(number/batch_size)
    print(run_times)
    all_time = 0.0
    with tf.Session(config = config, graph = graph) as sess:
        for t in range(int(run_times)):
            batch_images = image_data[t*batch_size:(t+1)*batch_size, ...]
            batch_images_name = all_images_name[t*batch_size:(t+1)*batch_size]
            batch_origin_images = all_images[t*batch_size:(t+1)*batch_size]
            start = time.time()
            print("batch_images shape:",batch_images.shape)
            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                    [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: batch_images})
            end = time.time()
            if t > 0:
                all_time = all_time + (end - start)
            single_pred_sbbox = np.split(pred_sbbox, batch_size, axis = 0)
            single_pred_mbbox = np.split(pred_mbbox, batch_size, axis = 0)
            single_pred_lbbox = np.split(pred_lbbox, batch_size, axis = 0)
            for i in range(batch_size):
                pred_bbox = np.concatenate(
                    [np.reshape(single_pred_sbbox[i], (-1, 5 + num_classes)),
                     np.reshape(single_pred_mbbox[i], (-1, 5 + num_classes)),
                     np.reshape(single_pred_lbbox[i], (-1, 5 + num_classes))],
                    axis=0)

                bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
                bboxes = utils.nms(bboxes, 0.45, method='nms')
                image = utils.draw_bbox(batch_origin_images[i], bboxes)
                image = Image.fromarray(image)
                img_path = "./result_img"
                if not (os.path.exists(img_path)):
                  os.mkdir(img_path)
                new_img_location = os.path.join(img_path, batch_images_name[i])
                image.save(new_img_location, 'jpeg')
        if run_times > 1:
          print('end2end fps: %f' %(((run_times-1) * batch_size)/all_time))
