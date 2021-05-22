import cv2
import math
import os
import numpy as np
import tensorflow as tf
from tensorflow.python import graph_util

tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', '/tmp/ch4_test_images/images/', '')
FLAGS = tf.app.flags.FLAGS

import model
def main(argv=None):
    import os
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
    output_file=os.path.join(FLAGS.output_dir, "./east.pb")
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        f_score, f_geometry = model.model(input_images, is_training=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

	mlu_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=mlu_config) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])
            with tf.gfile.FastGFile(output_file, mode = 'wb') as f:
              f.write(constant_graph.SerializeToString())
if __name__ == '__main__':
    tf.app.run()
