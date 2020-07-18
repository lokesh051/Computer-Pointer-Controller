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
import math

class GazeEstimation:
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


    def load_model(self):
        core = IECore()
        net = core.load_network(network=self.model, device_name=self.device, num_requests=1)
        supported_layers = core.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print('Unsupported layers found: {}'.format(unsupported_layers))
            exit(1)
            return

        self.input_name = [i for i in self.model.inputs.keys()]
        self.input_shape = self.model.inputs[self.input_name[1]].shape
        self.output_name = [o for o in self.model.outputs.keys()]
        #self.output_shape= self.model.outputs[self.output_name[1]].shape
        return net

    def wait(self, net):
        status = net.requests[0].wait(-1)
        return status

    def predict(self, net, left_image, right_image, pose_angles):
        res = []
        result= []
        if left_image is not None and right_image is not None and pose_angles is not None:
            left_eye_image = self.preprocess_input(left_image)
            right_eye_image = self.preprocess_input(right_image)
            input_dict={'left_eye_image': left_eye_image,'right_eye_image': right_eye_image,'head_pose_angles': pose_angles}
            net.infer(input_dict)
            if self.wait(net) == 0:
                result, res = self.preprocess_output(net, pose_angles)

        return res, result

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        try:
            input_img=cv2.resize(image, (self.input_shape[3],self.input_shape[2]))
            input_img=input_img.transpose((2,0,1))
            input_img=input_img.reshape(1, *input_img.shape)
        except:
            pass
        return input_img

    def preprocess_output(self, net, pose_angles):
         gaze_vector =  net.requests[0].outputs[self.output_name[0]][0]
         mouse_pointer = (0,0)
         try:
            angle_r_fc = pose_angles[2]
            sin_r = math.sin(angle_r_fc * math.pi / 180.0)
            cos_r = math.cos(angle_r_fc * math.pi / 180.0)
            x = gaze_vector[0] * cos_r + gaze_vector[1] * sin_r
            y = -gaze_vector[0] * sin_r + gaze_vector[1] * cos_r
            mouse_pointer = (x, y)
         except:
           pass

         return mouse_pointer, gaze_vector
