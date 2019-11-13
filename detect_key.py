#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import argparse
from models.yolo import YOLO, detect_video
from PIL import Image

import models.model_creation as mc

def detect_img(yolo, image_path, output_path=''):
    try:
        image = Image.open(image_path)
    except:
        print ('Open Error! Try again!')
    r_image = yolo.detect_image(image)
    r_image.save(output_path)
    yolo.close_session()


FLAGS = None


# class YOLO defines the default value, so suppress any default here

if __name__ == "__main__":

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('--gpu_num', type=int,
                        help='Number of GPU to use, default '
                        + str(YOLO.get_defaults('gpu_num')))
    parser.add_argument('--image', default=False, action='store_true',
                        help='Image detection mode, will ignore all positional arguments'
                        )
    parser.add_argument('--video', default=False, action='store_true',
                        help='Video detection mode, will ignore all positional arguments'
                        )
    parser.add_argument('--input', nargs='?', type=str,
                        help='Video or image input path')
    parser.add_argument('--output', nargs='?', type=str, default='',
                        help='[Optional] Image or Video output path')

    FLAGS = parser.parse_args()

    settings = mc.load_settings()
    classes = settings["classes"]
    anchors = settings["anchors"]
    model_path = 'model_data/' + settings['model_name'] + '.h5'

    if FLAGS.image:
        print ('Image detection mode')
        if 'input' in FLAGS:
            detect_img(YOLO(classes, anchors, model_path), FLAGS.input, FLAGS.output)
        else:
            print ('Must specify at least image_input_path.  See usage with --help.')
    elif FLAGS.video:
        if 'input' in FLAGS:
            detect_video(YOLO(classes, anchors, model_path), FLAGS.input, FLAGS.output)
        else:
            print ('Must specify at least video_input_path.  See usage with --help.')

