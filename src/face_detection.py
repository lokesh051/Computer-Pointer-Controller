'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore, IEPlugin
import os
import cv2
import argparse
import sys

class Face_Model:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, threshold, device='CPU',extensions=None):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.input_blob = None
        self.output_blob = None

        self.model=IENetwork(self.model_structure, self.model_weights)
        self.input_name= None
        self.input_shape= None
        self.output_name= None
        self.output_shape= None

    def crop_image(self, frame_copy, result, threshold, width, height):
        crop_img = None
        coords = []
        for box in result[0][0]: # Output shape is 1x1x100x7
            conf = box[2]
            if conf >= threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                crop_img = frame_copy[ymin:ymax, xmin:xmax]
                coords = [xmin, ymin, xmax, ymax]

        return crop_img, coords


    def load_model(self):
        core = IECore()
        net = core.load_network(network=self.model, device_name=self.device, num_requests=1)
        supported_layers = core.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print('Unsupported layers found: {}'.format(unsupported_layers))
            exit(1)
            return
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        return net

    def wait(self, net):
        status = net.requests[0].wait(-1)
        return status

    def predict(self, image, net, width, height):
        self.input_blob=next(iter(self.model.inputs))
        self.output_blob = next(iter(self.model.outputs))
        frame_copy = image.copy()
        frame = self.preprocess_input(image)
        input_dict={self.input_blob:frame}
        net.infer(input_dict)
        result = []
        if self.wait(net) == 0:
            result = self.preprocess_output(net)
            out_frame, coords = self.crop_image(frame_copy, result, self.threshold , width, height)

        return out_frame, coords

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        input_img=cv2.resize(image, (self.input_shape[3],self.input_shape[2]))
        input_img=input_img.transpose((2,0,1))
        input_img=input_img.reshape(1, *input_img.shape)
        return input_img

    def preprocess_output(self, net):
         return net.requests[0].outputs[self.output_blob]
