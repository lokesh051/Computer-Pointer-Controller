'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2
from numpy import ndarray

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
    
    def load_data(self):
        if self.input_type=='video':
            self.input_stream = self.input_file
            self.cap=cv2.VideoCapture(self.input_stream)
        elif self.input_type=='cam':
            self.input_stream = 0
            self.cap=cv2.VideoCapture(0)
        else:
            self.input_stream = self.input_file
            self.cap=cv2.imread(self.input_stream)

        self.cap.open(self.input_stream)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        while self.cap.isOpened():
            flag, frame=self.cap.read()
            if not flag:
                break

            if frame is not None:
                cv2.imshow("Video Input", frame)

            yield frame

            key_pressed = cv2.waitKey(60)


            if key_pressed == 27:
                break


    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()

