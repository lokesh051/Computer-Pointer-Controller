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

class FacialLandmark:
    '''
        Class for the Head Pose Detection Model.
        '''
    def __init__(self, model_name, threshold=0.6, device='CPU',extensions=None):
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

    def predict(self, image, net):
        self.input_blob=next(iter(self.model.inputs))
        self.output_blob = next(iter(self.model.outputs))
        frame = self.preprocess_input(image)
        input_dict={self.input_blob:frame}
        net.infer(input_dict)
        if self.wait(net) == 0:
            left_eye_image, right_eye_image, eye_cords = self.preprocess_output(net, image)

        return left_eye_image, right_eye_image, eye_cords

    def wait(self, net):
        status = net.requests[0].wait(-1)
        return status


    def check_model(self):
        pass

    def preprocess_input(self, image):
        input_img=cv2.resize(image, (self.input_shape[3],self.input_shape[2]))
        input_img=input_img.transpose((2,0,1))
        input_img=input_img.reshape(1, *input_img.shape)
        return input_img

    def preprocess_output(self, net, image):
        left_eye_image, right_eye_image, eye_cords = [], [], []
        try:
            landmarks =  net.requests[0].outputs[self.output_blob][0]
            h = image.shape[0]
            w = image.shape[1]
            left_eye_xmin = int(landmarks[0][0][0] * w) - 10
            left_eye_ymin = int(landmarks[1][0][0] * h) - 10
            right_eye_xmin = int(landmarks[2][0][0] * w) - 10
            right_eye_ymin = int(landmarks[3][0][0] * h) - 10

            left_eye_xmax = int(landmarks[0][0][0] * w) + 10
            left_eye_ymax = int(landmarks[1][0][0] * h) + 10
            right_eye_xmax = int(landmarks[2][0][0] * w) + 10
            right_eye_ymax = int(landmarks[3][0][0] * h) + 10
            left_eye_image = image[left_eye_ymin:left_eye_ymax, left_eye_xmin:left_eye_xmax]
            right_eye_image = image[right_eye_ymin:right_eye_ymax, right_eye_xmin:right_eye_xmax]
            eye_cords = [[left_eye_xmin, left_eye_ymin, left_eye_xmax, left_eye_ymax],
                         [right_eye_xmin, right_eye_ymin, right_eye_xmax, right_eye_ymax]]
        except:
            pass


        return left_eye_image, right_eye_image, eye_cords
