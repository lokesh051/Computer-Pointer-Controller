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

class HeadPose:
    '''
        Class for the Head Pose Detection Model.
        '''
    def __init__(self, model_name,threshold, device='CPU',extensions=None):
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
        out_image = image
        if self.wait(net) == 0:
            yaw, pitch, roll = self.preprocess_output(net)

        return yaw, pitch, roll

    def wait(self, net):
        status = net.requests[0].wait(-1)
        return status

    
    def check_model(self):
        raise NotImplementedError
    
    def preprocess_input(self, image):
        input_img=cv2.resize(image, (self.input_shape[3],self.input_shape[2]))
        input_img=input_img.transpose((2,0,1))
        input_img=input_img.reshape(1, *input_img.shape)
        return input_img
    
    def preprocess_output(self, net):
        yaw = net.requests[0].outputs['angle_y_fc'][0][0]
        pitch = net.requests[0].outputs['angle_p_fc'][0][0]
        roll = net.requests[0].outputs['angle_r_fc'][0][0]
        return yaw, pitch, roll
