# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import logging
import os
import io
from tqdm import tqdm
import shutil

from lxml import etree
import PIL.Image
import PIL.ImageDraw
import tensorflow.compat.v1 as tf
import xml.etree.ElementTree as ET

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string(
    'images_dir', '', 'Root directory to folder containing image files.')
flags.DEFINE_string('annotations_dir', '',
                    'folder containing all xml annotations.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string(
    'label_map_path', 'data/detrac_label_map.pbtxt', 'Path to label map proto')

FLAGS = flags.FLAGS

def image_to_byte_array(image:PIL.Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format=image.format)
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr

def dict_to_tf_example(image_path,
                       annotations_xml,
                       label_map_dict,
                       ignore_regions_xml=[],
                       video_name=""):

    image_name = video_name + '_' +  os.path.basename(image_path)
    image = PIL.Image.open(image_path)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    new_img = PIL.ImageDraw.Draw(image)
    for black in ignore_regions_xml:
        x1 = float(black.get('left'))
        y1 = float(black.get('top'))
        w = float(black.get('width'))
        h = float(black.get('height'))
        x2 = x1 + w
        y2 = y1 + h
        new_img.rectangle([(int(x1), int(y1)), (int(x2), int(y2))], fill="#000000")

    image_bytes = image_to_byte_array(image)

    key = hashlib.sha256(image_bytes).hexdigest()

    width, height = image.size

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    re_ids = []
    for obj in annotations_xml:
        box = obj[0]
        obj_name = obj[1].get('vehicle_type')
        re_id = obj.get('id')
        x1 = float(box.get('left'))
        y1 = float(box.get('top'))
        w = float(box.get('width'))
        h = float(box.get('height'))
        x2 = x1 + w
        y2 = y1 + h
        xmin.append(float(x1) / width)
        ymin.append(float(y1) / height)
        xmax.append(float(x2) / width)
        ymax.append(float(y2) / height)
        classes_text.append(obj_name.encode('utf8'))
        classes.append(label_map_dict[obj_name])
        re_ids.append(int(re_id))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            image_name.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            image_name.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(image_bytes),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/re_id': dataset_util.int64_list_feature(re_ids),
    }))
    return example


def main(_):

    data_dir = FLAGS.images_dir
    data_new_dist = '/home/ubuntu/ibrahim/master/datasets/sub/train/'

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    logging.info('Reading Dataset from %s.', data_dir)
    for directory in tqdm(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, directory)):
            annotation_xml = os.path.join(
                FLAGS.annotations_dir, directory+".xml")
            logging.info('Parsing annotation %s.', annotation_xml)
            xml = ET.parse(annotation_xml)
            ignored_region = xml.find('ignored_region')
            if not os.path.exists(os.path.join(data_new_dist, directory)):
                os.makedirs(os.path.join(data_new_dist, directory))
            c = 0
            for image in tqdm(sorted(os.listdir(os.path.join(data_dir, directory)))):
                if c>=5:
                    image_path = os.path.join(data_dir, directory, image)
                    dst_path = os.path.join(data_new_dist, directory, image)
                    shutil.copyfile(image_path, dst_path)
                    frame_no = int(image.replace('img', '').replace('.jpg', ''))
                    target_list = xml.find(f".//*[@num='{frame_no}']")
                    annotations = target_list[0] if target_list else []
                    tf_example = dict_to_tf_example(
                        dst_path, annotations, label_map_dict, ignored_region, directory)
                    writer.write(tf_example.SerializeToString())
                    c=0
                c+=1
    writer.close()


if __name__ == '__main__':
    tf.app.run()
