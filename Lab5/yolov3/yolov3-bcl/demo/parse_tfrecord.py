from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
def read_and_decode(filename):
  filename_queue = tf.train.string_input_producer([filename])
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example, features={
                                                          'image/height':tf.FixedLenFeature([], tf.int64),
                                                          'image/width':tf.FixedLenFeature([],tf.int64),
                                                          'image/filename':tf.FixedLenFeature([], tf.string),
                                                          'image/source_id':tf.FixedLenFeature([], tf.string),
                                                          'image/key/sha256':tf.FixedLenFeature([], tf.string),
                                                          'image/encoded':tf.FixedLenFeature([], tf.string),
                                                          'image/format':tf.FixedLenFeature([], tf.string),
                                                          'image/object/bbox/xmin':tf.VarLenFeature(tf.float32),
                                                          'image/object/bbox/xmax':tf.VarLenFeature(tf.float32),
                                                          'image/object/bbox/ymin':tf.VarLenFeature(tf.float32),
                                                          'image/object/bbox/ymax':tf.VarLenFeature(tf.float32),
                                                          'image/object/class/text':tf.VarLenFeature(tf.string),
                                                          'image/object/class/label':tf.VarLenFeature(tf.int64),
                                                          'image/object/difficult':tf.VarLenFeature(tf.int64),
                                                          'image/object/group_of':tf.VarLenFeature(tf.int64),
                                                          'image/object/truncated':tf.VarLenFeature(tf.int64),
                                                          'image/object/view':tf.VarLenFeature(tf.string),
  })

  with tf.device('/cpu:0'):
    h = tf.cast(features['image/height'], tf.int32)
    w = tf.cast(features['image/width'], tf.int32)
    format = features['image/format']
    encoded = features['image/encoded']
    xmin = features['image/object/bbox/xmin']
    xmax = features['image/object/bbox/xmax']
    ymin = features['image/object/bbox/ymin']
    ymax = features['image/object/bbox/ymax']
    label = features['image/object/class/label']
    difficult = features['image/object/difficult']
    group_of = features['image/object/group_of']
    truncated = features['image/object/truncated']
    img_name = features['image/filename']
  return img_name, label, xmin, xmax, ymin, ymax, difficult, group_of, truncated, h, w
  #return img_name, xmin, ymin, xmax, ymax, label

def get_sample_counts(filename):
  count = 0
  for record in tf.python_io.tf_record_iterator(path=filename):
    count += 1
  return count

def sparse_to_list(sparses, count):
  print("--------------------------------------")
  list = []
  indices = sparses.indices
  values = sparses.values
  cursor = 0
  for i in range(count):
    index = indices[indices[:,0] == i][:,1]
    index += cursor
    list.append(values[index].tolist())
    cursor += len(index)
  return list

def convert_label(label_list):
  indince = [12,26,29,30,45,66,68,69,71,83]
  reverse_indince = indince[::-1]
  print("aaaaaaaaa", reverse_indince)
  for i in range(len(label_list)):
    for j in range(len(indince)):
      if (label_list[i] >= reverse_indince[j]):
        print ( "i j",i,j)
        print("=====",indince[9-j])
        print("=====",indince.index(indince[9-j]))
        label_list[i] = label_list[i] - indince.index(indince[9-j]) - 1
        break
  return label_list




if __name__ == "__main__":
  records = "./tf_records"
  batch_size = 1
  number = 8
  parser = argparse.ArgumentParser()
  parser.add_argument("--records", help="records to be processed")
  parser.add_argument("--batch_size", help="batch_size to be processed")
  parser.add_argument("--number", type=int, help="number of records to be processed")
  args = parser.parse_args()
  if args.records:
    records = args.records
  if args.batch_size:
    batch_size = int(args.batch_size)
  if args.number:
    number = args.number
  print("records =================================",records)
  #name, xmin, ymin, xmax, ymax, label = read_and_decode(records)
  name, label, xmin, xmax, ymin, ymax, difficult, group_of, truncated ,height, width = read_and_decode(records)
  #name_batch, xmin_batch, ymin_batch, xmax_batch, \
  #  ymax_batch, label_batch = tf.train.batch([name, xmin, ymin, xmax,
  #  ymax, label], batch_size=batch_size)
  name_batch, label_batch, xmin_batch, xmax_batch, \
      ymin_batch, ymax_batch, difficult_batch, group_of_batch, height_batch, width_batch = tf.train.batch([
                                                                        name, label, xmin, xmax,
                                                                        ymin, ymax, difficult, group_of,
                                                                        height, width], batch_size=batch_size)

  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  config = tf.ConfigProto(allow_soft_placement=True,
			  inter_op_parallelism_threads=1,
                          intra_op_parallelism_threads=1)

  sample_counts = get_sample_counts(records)
  print("tfrecord sample_counts = ", sample_counts)
  if number > sample_counts:
    number = sample_counts
  if number < batch_size:
    print("Error! number of images must be >= batch_size")
    exit(0)

  print("batch_size =", batch_size)
  print("iter =", int(number/batch_size))
  with tf.Session( config = config) as session:
    with open('./coco_val.txt', 'wa') as f:
      for j in range(int(number/batch_size)):
          #name, labels, xmins, xmaxes, ymins, ymaxes= sess.run([name_batch,
          #                                                      label_batch,
          #                                                      xmin_batch,
          #                                                      xmax_batch,
          #                                                      ymin_batch,
          #                                                      ymax_batch])
          name, labels, xmins, xmaxes, ymins, ymaxes, difficults, group_ofs, heights, widths = sess.run([name_batch,
                                                                                                         label_batch,
                                                                                                         xmin_batch,
                                                                                                         xmax_batch,
                                                                                                         ymin_batch,
                                                                                                         ymax_batch,
                                                                                                         difficult_batch,
                                                                                                         group_of_batch,
                                                                                                         height_batch,
                                                                                                         width_batch])

          labels_list = np.array(sparse_to_list(labels, batch_size)).reshape(-1)
          #print("============================")
          xmins_list = sparse_to_list(xmins, batch_size)
          print ("aaaaaaa xmin ",xmins)
          #print("xxxxxxxxxxxx xmins_list trype",type(xmins_list))
          #print("xxxxxxxxxxxx xlabels_list trype",type(labels_list))
          ymins_list = sparse_to_list(ymins, batch_size)
          xmaxes_list = sparse_to_list(xmaxes, batch_size)
          ymaxes_list = sparse_to_list(ymaxes, batch_size)

          # xmin.append(float(x) / image_width)
          # xmax.append(float(x + bbox_width) / image_width)
          # ymin.append(float(y) / image_height)
          # ymax.append(float(y + bbox_height) / image_height)
          # convert  normalize location to real location
          real_xmin = (xmins_list * widths).reshape(-1)
          #print("xxxxxxxxxxxx xmins_list trype",type(real_xmin))
          real_xmax = (xmaxes_list * widths).reshape(-1)
          real_ymin = (ymins_list * heights).reshape(-1)
          real_ymax = (ymaxes_list * heights).reshape(-1)


          print ("file name ", name)
          print ("xmin ",xmins_list)
          print ("xmax ",xmaxes_list)
          print ("ymin ",ymins_list)
          print ("ymax ",ymaxes_list)
          print ("height ",heights)
          print ("weight ", widths)
          print ("label ",labels_list)
          new_labels_list = convert_label(labels_list)
          print ("new label ", new_labels_list)
          print ("real_xmin ",real_xmin)
          #print ("real_xmin_0 ",real_xmin[0])
          print ("real_xmax ",real_xmax)
          print ("real_ymin ",real_ymin)
          print ("real_ymax ",real_ymax)
          print ("iter = " ,j)
          if(len(real_xmin.tolist())==0):
              continue
          else:
              f.write("./data/dataset/%s "% name[0])
          if (len(labels_list) != 0):
              for i in range(len(labels_list)):
                  f.write("%d,%d,%d,%d,%d " % (real_xmin[i],real_ymin[i],real_xmax[i],real_ymax[i],new_labels_list[i]))
              f.write('\n')
    f.close()


  coord.request_stop()
  coord.join(threads)
  sess.close()











