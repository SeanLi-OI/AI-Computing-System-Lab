# Copyright (C) [2020] by Cambricon, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall self.tcp included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# pylint: disable=invalid-name
"""Non-maximum suppression operator implementation using BANGPy TCP API."""
import os, shutil
import numpy as np
import bangpy as bp
from bangpy import tcp, load_module
from bangpy.tcp.build import TaskType
from bangpy import enabled_targets
TARGET = enabled_targets()

NMS_SIZE = 64

def PAD_UP(x, y):
    return ((x -1)/ y + 1) * y

def PAD_DOWN(x, y):
    return x / y * y

def _py_nms(output, iou_threshold=0.5, score_threshold=0.5, valid_num=1):
    def _iou(boxA, boxB):
        """Compute the Intersection over Union between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Compute the area of both the prediction and ground-truth rectangle
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Compute the intersection over union
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    assert(len(output) != 0), "box num must self.tcp valid!"
    output = output[np.argsort(-output[:, 0], kind="stable")]
    bboxes = [output[0]]

    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j, _ in enumerate(bboxes):
            if _iou(bbox[1:5], bboxes[j][1:5]) >= iou_threshold:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)

    bboxes = np.asarray(bboxes, np.float)
    score_mask = bboxes[:, 0] > score_threshold
    bboxes = bboxes[score_mask, :]

    if len(bboxes) >= valid_num:
        return_value = bboxes[:valid_num]

    else:
        zeros = np.zeros((valid_num - len(bboxes), 5))
        return_value = np.vstack([bboxes, zeros])

    return return_value

class NMS(object):
    """Operator description.
        Non_maximum_suppression operator for object detection.

    Parameters
    ----------
    task_num : Int
        MLU task number.

    name : Str, optional
        The name of operator.
    """
    def __init__(self, dtype=bp.float16, name="nms"):
        self.dtype = dtype
        self.name = name
        self.tcp = tcp.TCP(TARGET)
        self.splitNum = self.tcp.Var("splitNum")
        self.class_start = self.tcp.Var("classStart")
        self.class_end = self.tcp.Var("classEnd")
        self.class_num = self.tcp.Var("classNum")
        self.num_entries = self.tcp.Var("num_entries")
        self.nmsBoxCount = self.tcp.Var("nmsBoxCount")
        self.batchIdx = self.tcp.Var("batchIdx")
        self.taskid = self.tcp.Var("TASKIDX_temp")
        self.buffer_size = self.tcp.Var("buffer_size")
        self.num_boxes = self.tcp.Var("input_box_num")
        self.max_output_size = self.tcp.Var("keepNum")
        self.input_stride = self.tcp.Var("input_stride")
        self.output_stride = self.tcp.Var("output_stride")
        self.iou_threshold = self.tcp.Var("thresh_iou", dtype=bp.float16)
        self.score_threshold = self.tcp.Var("thresh_score", dtype=bp.float16)
        self.gdram_save_count = 256
        self.buffer_segment = 11

#the code need student to upload.
#/**********************************************************************************************************************************/
    #output:[5, output_stride]
    #input_score:[input_stride,]
    #input_box:[4, input_stride]
    #buffer_nram:[buffer_size]
    def nms_compute_body(self, output, input_score, input_box, buffer_nram):
        """The main compute body of the nms operator."""
        self.output = output
        self.input_box = input_box
        self.input_score = input_score
        self.buffer = buffer_nram

        self.max_score = self.tcp.Tensor(shape=[64], dtype=self.dtype,
                                         name="max_score", scope="nram")

        # Calculate the maximum tensor size according to the tensor involved in nms's computation
        nram_size_limit = ...
        self.max_seg_size = self.tcp.Scalar(dtype=bp.int32, name="max_seg_size",
                                            value=PAD_DOWN(nram_size_limit, NMS_SIZE))


        self.score = self.buffer[...]
        self.x1 = self.buffer[...]
        self.y1 = self.buffer[...]
        self.x2 = self.buffer[...]
        self.y2 = self.buffer[...]
        self.inter_x1 = self.buffer[...]
        self.inter_y1 = self.buffer[...]
        self.inter_x2 = self.buffer[...]
        self.inter_y2 = self.buffer[...]
        self.max_box = self.buffer[...]
        self.nram_save = self.buffer[...]
        self.max_pos = self.tcp.Tensor(shape=[64], dtype=bp.int32, name="max_pos",
                                       scope="nram")

        #Scalar declaration
        #how to explain Var as quote?
        # self.output_box_num = self.tcp.Tensor(shape=[1,], dtype=bp.int32, name="output_box_num", scope="nram", explicit_alloc=False)
        with self.tcp.if_scope(self.output_box_num != 0):
            self.output_box_num.assign(0)
        self.nram_save_count = self.tcp.Scalar(dtype=bp.int32, name="nram_save_count", value=0)
        self.save_time = self.tcp.Scalar(dtype=bp.int32, name="save_time", value=0)
        self.zero_scalar = self.tcp.Scalar(dtype=self.dtype, name="zero_scalar", value=0)

        core_num = self.tcp.Scalar(dtype=bp.int32, name="core_num",
                                   value=self.num_boxes)
        copy_begin = self.tcp.Scalar(dtype=bp.int32, name="copy_begin", value=0)

        input_offset = self.tcp.Scalar(dtype=bp.int32, name="input_offset", value=copy_begin)


        repeat = self.tcp.Scalar(dtype=bp.int32, name="repeat", value=core_num / self.max_seg_size)
        remain = self.tcp.Scalar(dtype=bp.int32, name="remain", value=core_num % self.max_seg_size)
        remain_pad = self.tcp.Scalar(dtype=bp.int32, name="remain_pad",
                                     value=PAD_UP(remain, NMS_SIZE))
        stop_tag = self.tcp.Scalar(dtype=bp.int32, name="stop_tag", value=0)
        with self.tcp.for_range(0, self.max_output_size) as _:
            with self.tcp.if_scope(stop_tag != -1):
                self.max_box[0] = self.zero_scalar
                # Look for the max score and its corresponding index.
                max_index = self.score_sort(...)

                # the max box's x1, y1, x2, y2 on every core
                self.tcp.memcpy(...)
                self.tcp.memcpy(...)
                self.tcp.memcpy(...)
                self.tcp.memcpy(...)

                max_area = ...
                self.input_score[max_index] = 0
                global_max_index = max_index
                self.max_score[0] = self.max_box[0]

                # by now, we get: max_score|max_index|max_box|max_area
                with self.tcp.if_scope(self.output_box_num != 0):
                    with self.tcp.if_scope(tcp.any(self.nram_save_count == self.gdram_save_count,
                                                self.max_score[0] <= self.score_threshold)):
                        # score, x1, y1, x2, y2
                        self.tcp.memcpy(
                            self.output[....],
                            self.nram_save[...]
                        )
                        self.save_time.assign(...)
                        self.nram_save_count.assign(...)
                with self.tcp.if_scope(self.max_score[0] <= self.score_threshold):
                    stop_tag.assign(-1)
                with self.tcp.if_scope(self.max_score[0] > self.score_threshold):
                    # score, x1, y1, x2, y2
                    idx = ...
                    self.tcp.memcpy(
                        self.nram_save[....].reshape(...)[...],
                        self.max_box[...].reshape(...))

                    self.nram_save_count.assign(...)
                    self.output_box_num.assign(...)

                    with self.tcp.if_scope(...):
                        self.tcp.memcpy(
                            self.output[...],
                            self.nram_save[...]
                        )

                    self.score_rewrite(...)

    def score_sort(self, input_offset, nms_loop, remain, remain_pad):
        """Sort the boxes' score."""
        max_index = self.tcp.Scalar(dtype=bp.int32, name="input_box")
        with self.tcp.if_scope(...):
            with self.tcp.for_range(0, nms_loop) as i:
                offset = ....
                max_index.assign(self.score_sort_each_loop(...))

        offset = ...
        with self.tcp.if_scope(...):
            max_index.assign(self.score_sort_each_loop(...))
        return max_index

    def score_sort_each_loop(self, input_offset, offset, loop_extent, alignmemt):
        """Sort the boxes' score in each loop."""
        self.tcp.assign(self.score[...], 0)
        idx = ...
        self.tcp.memcpy(self.score[...], self.input_score[...])
        self.tcp.amax(self.inter_x1, self.score[...])

        with self.tcp.if_scope(...):
            self.max_box[0] = ...
            # offset start from head of input data
            max_index = idx + self.tcp.uint_reinterpret(self.inter_x1[...])
        return max_index

    def score_rewrite(self, max_area, input_offset, nms_loop, remain, remain_pad):
        """Rewrite the score of boxes."""
        offset = 0
        with self.tcp.if_scope(...):
            with self.tcp.for_range(...) as i:
                offset = ...
                self.score_rewrite_each_loop(...)

        offset = ...
        with self.tcp.if_scope(...):
            self.score_rewrite_each_loop(...)


    def score_rewrite_each_loop(self, max_area, input_offset, offset, loop_extent, alignmemt):
        """Rewrite the score of each loop."""
        self.tcp.assign(self.score[...], 0)
        idx = ...
        self.tcp.memcpy(self.score[:loop_extent], self.input_score[idx:idx+loop_extent])
        self.tcp.memcpy(self.x1[...], self.input_box[...])
        self.tcp.memcpy(self.y1[...], self.input_box[...])
        self.tcp.memcpy(self.x2[...], self.input_box[...])
        self.tcp.memcpy(self.y2[...], self.input_box[...])

        # 1、 compute IOU
        # get the area_I
        self.tcp.assign(self.inter_y1[...], self.max_box[1])
        self.tcp.maximum(self.inter_x1[...], self.x1[...],
                         self.inter_y1[...])
        self.tcp.assign(self.inter_y2[...], self.max_box[3])
        self.tcp.minimum(self.inter_x2[...], self.x2[...],
                         self.inter_y2[...])
        self.tcp.subtract(self.inter_x1[...], self.inter_x2[...],
                          self.inter_x1[...])
        self.tcp.relu(self.inter_x1[...], self.inter_x1[...])
        self.tcp.assign(self.inter_x2[], self.max_box[2])
        self.tcp.maximum(self.inter_y1[], self.y1[],
                         self.inter_x2[])
        self.tcp.assign(self.inter_x2[], self.max_box[4])
        self.tcp.minimum(self.inter_y2[], self.y2[],
                         self.inter_x2[])
        self.tcp.subtract(self.inter_y1[], self.inter_y2[],
                          self.inter_y1[])
        self.tcp.relu(self.inter_y1[], self.inter_y1[])

        self.tcp.multiply(self.inter_x1[...], self.inter_x1[...],
                          self.inter_y1[...])
        # get the area of input_box: area = (self.x2 - self.x1) * (y2 - self.y1)
        self.tcp.subtract(self.inter_y1[...], self.x2[...], self.x1[...])
        self.tcp.subtract(self.inter_y2[...], self.y2[...], self.y1[...])
        self.tcp.multiply(self.inter_x2[...], self.inter_y1[...],
                          self.inter_y2[...])

        # get the area_U: area + max_area - area_I..
        self.tcp.assign(self.inter_y1[.], max_area)
        self.tcp.add(self.inter_x2[...], self.inter_x2[...],
                     self.inter_y1[...])
        self.tcp.subtract(self.inter_x2[...], self.inter_x2[...],
                          self.inter_x1[...])

        # 2、 select the box
        # if IOU greater than thres, set the score to zero, abort it: area_U * thresh > area_I?
        self.tcp.multiply(self.inter_x2[...], self.inter_x2[...], ...)
        self.tcp.greater(self.inter_x1[...], self.inter_x2[...],
                         self.inter_x1[...])
        self.tcp.multiply(self.score[...], self.score[...], self.inter_x1[...])

        # update the score
        idx = ...
        self.tcp.memcpy(self.input_score[...], self.score[...])
#/**********************************************************************************************************************************/

    def nms_compute(self):
        """The entry of the nms operator."""
        self.output_gdram = self.tcp.Tensor(shape=[self.class_num * self.output_stride], dtype=self.dtype,
                                           name="gdram_score", scope="global")

        self.input_gdram = self.tcp.Tensor(
            shape=[4 * self.splitNum ,self.num_entries, 2048], dtype=self.dtype,
            name="input_data_box", scope="global")

        self.boxCounts = self.tcp.Tensor(
            shape=[4,], dtype=bp.int32,
            name="boxCounts_nram", scope="global")

        self.nmsBoxCount = self.tcp.Tensor(
            shape=[1,], dtype=bp.int32,
            name="nmsBoxCount", scope="global")

        self.full_buffer = self.tcp.Tensor(shape=[(self.tcp.get_ram_size("nram") -1024) // self.dtype.bytes],
                                      dtype=self.dtype, name="buffer", scope="nram")

        self.part = self.tcp.Scalar(dtype=bp.int32, name="part", value=0)
        with self.tcp.for_range(0, self.splitNum, name="coreIdx") as i:
            with self.tcp.if_scope(self.boxCounts[i] > 0):
                self.tcp.memcpy(self.full_buffer[self.part:][:4 * self.input_stride].reshape([4, self.input_stride])[:,:self.boxCounts[i]],
                                self.input_gdram[self.taskid + i,:4,:self.boxCounts[i]])
                self.part.assign(self.boxCounts[i]+ self.part)

        self.current_class_num = self.tcp.Scalar(dtype=bp.int32, name="currClassNum", value=0)
        self.output_box_num = self.tcp.Scalar(dtype=bp.int32, name="output_box_num")
        min_temp_a = self.tcp.Scalar(dtype=bp.int32, name="temp_a")
        min_temp_b = self.tcp.Scalar(dtype=bp.int32, name="temp_b")        
        with self.tcp.for_range(0, self.class_end - self.class_start) as i:
            with self.tcp.if_scope(self.current_class_num == 0):
                self.part.assign(0)
                self.tcp.assign(self.full_buffer[4*self.input_stride:][:self.class_num*self.input_stride], -999)
                with self.tcp.for_range(0, self.splitNum, name="coreIdx") as j:
                    min_temp_a.assign(self.class_num - 1)
                    min_temp_b.assign(self.class_end - self.class_start -i)
                    class_num_loop = self.tcp.scalar_min(min_temp_a, min_temp_b) + 1
                    with self.tcp.if_scope(self.boxCounts[j] > 0):
                        self.tcp.memcpy(self.full_buffer[4 * self.input_stride + self.part:][:class_num_loop * self.input_stride].reshape([class_num_loop, self.input_stride])[:,:self.boxCounts[j]],
                                        self.input_gdram[self.taskid + j,i+self.class_start+5:i+self.class_start+5+class_num_loop,:self.boxCounts[j]])
                        self.part.assign(self.boxCounts[j]+ self.part)
                self.current_class_num.assign(self.class_num)

            self.output_box_num.assign(0)
            self.nms_compute_body(
                self.output_gdram[2*self.output_stride + self.nmsBoxCount[0]:][:5*self.output_stride].reshape([5,self.output_stride]),
                self.full_buffer[(4 + self.class_num-self.current_class_num)* self.input_stride:][:self.input_stride],
                self.full_buffer[: 4*self.input_stride].reshape([4, self.input_stride]),
                self.full_buffer[(4+self.class_num)*self.input_stride:]
            )

            with self.tcp.if_scope(self.output_box_num > 0):
                self.tcp.assign(self.full_buffer[(4 + self.class_num) * self.input_stride:][:self.input_stride], self.batchIdx)
                self.tcp.memcpy(
                    self.output_gdram[self.nmsBoxCount[0]:][:self.output_box_num],
                    self.full_buffer[(4 + self.class_num) * self.input_stride:][:self.output_box_num]
                )
                self.tcp.assign(self.full_buffer[(4 + self.class_num) * self.input_stride:][:self.input_stride], self.class_start + i)
                self.tcp.memcpy(
                    self.output_gdram[self.nmsBoxCount[0] + self.output_stride:][:self.output_box_num],
                    self.full_buffer[(4 + self.class_num) * self.input_stride:][:self.output_box_num]
                )
            self.current_class_num.assign(self.current_class_num - 1)
            self.nmsBoxCount[0] += self.output_box_num       

        return self.tcp.BuildBANG(inputs=[self.input_gdram, self.boxCounts, self.splitNum, self.class_end,
                                          self.class_num, self.class_start, self.nmsBoxCount, self.batchIdx, self.taskid,
                                          self.num_entries, self.num_boxes, self.max_output_size, self.iou_threshold,
                                          self.score_threshold, self.buffer_size, self.input_stride, self.output_stride],
                                  outputs=[self.output_gdram], kernel_name=self.name)

def verify_nms():
    """Test nms operator by giving multiple sets of parameters."""
    nms_operator = NMS(dtype=bp.float16)
    fnms = nms_operator.nms_compute()
    if os.path.exists('./test'):
        shutil.rmtree('./test')
    fnms.save('./test/')
    new_line = []
    with open('./test/device.mlu', "r+") as f:
        line_num = 0
        for line in f:
            if line_num == 0:
                new_line.append(
                    "__mlu_func__ void nms_detection( int* boxCounts_nram,  half* input_data_box,  half* gdram_score,  int* nmsBoxCount, int splitNum, int num_entries, int TASKIDX_temp, int input_stride, " + \
                    "int classEnd, int classStart, int classNum, int buffer_size, int input_box_num, int keepNum, half thresh_score, int output_stride, half thresh_iou, int batchIdx, half* buffer) {\n"
                )
                line_num = 1
            else:
                if line.find("__nram__ half buffer") == -1:
                    new_line.append(line)
    with open('./test/device.mlu', "w+") as f:
        f.writelines(new_line)    
    os.system('cp ./test/device.mlu nms_detection.h')

if __name__ == "__main__":
    verify_nms()
