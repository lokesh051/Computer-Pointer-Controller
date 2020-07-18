from argparse import ArgumentParser

import cv2
from face_detection import Face_Model
from facial_landmarks_detection import FacialLandmark
from gaze_estimation import GazeEstimation
from head_pose_estimation import HeadPose
from mouse_controller import MouseController
import os
import logging
import time


def main(args):
    model=args.faceModel
    headpose = args.headpose
    device=args.device
    facialLandmark = args.facialLandmark
    gazeEstimation = args.gazeEstimation
    input_arg = args.input
    threshold = float(args.threshold)
    logging.basicConfig(filename='error_log.log', filemode='w')
    error_log = logging.getLogger()

    if input_arg == 'cam':
        input_stream = 0
        cap=cv2.VideoCapture(input_stream)
    else:
        if os.path.isfile(input_arg):
            input_stream = input_arg
            cap=cv2.VideoCapture(input_stream)
        else:
            print('Could not determine the file location or Could not load the desired format, please use .mp4 or cam')
            error_log.error('Could not determine the file location or Could not load the desired format, please use .mp4 or cam')
            exit(1)
            return

    model_load_time = time.time()

    # Load Face Detection Model
    face_detection = Face_Model(model, threshold, device=device)
    face_net = face_detection.load_model()

    # Load Head Pose Detection Model
    head_pose = HeadPose(headpose, threshold, device=device)
    head_net =  head_pose.load_model()

    # Load Facial Landmarks Model
    facial_landmarks = FacialLandmark(facialLandmark,threshold, device=device)
    landmark_net = facial_landmarks.load_model()

     # Load Gaze Estimation Model
    gaze_estimation = GazeEstimation(gazeEstimation, threshold, device=device)
    gaze_net = gaze_estimation.load_model()

    total_loading_time = time.time() - model_load_time

    mouse_controller = MouseController('medium', 'slow')


    width = int(cap.get(3))
    height = int(cap.get(4))

    out_video = cv2.VideoWriter('out_video.mp4', cv2.VideoWriter_fourcc(*'avc1'), int(cv2.CAP_PROP_FPS), (width,height), True)
    frame_count = 0
    inference_time = time.time()
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        flag, frame=cap.read()
        if not flag:
            break



        key_pressed = cv2.waitKey(60)

        out_frame, face_coords = face_detection.predict(frame, face_net, width, height)
        if out_frame is not None:
            if not (out_frame.shape[1] == 0 or out_frame.shape[0] == 0):
                yaw, pitch, roll = head_pose.predict(out_frame, head_net)
                head_pose_angles = [yaw, pitch, roll]
                left_eye_image, right_eye_image, eye_cords = facial_landmarks.predict(out_frame, landmark_net)
                mouse_pointer, gaze_vector = gaze_estimation.predict(gaze_net, left_eye_image, right_eye_image, head_pose_angles)



                mouse_controller.move(-mouse_pointer[0], mouse_pointer[1])
                if frame_count % 5 == 0:
                    cv2.putText(frame, "Head Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(head_pose_angles[0],head_pose_angles[1],head_pose_angles[2]), (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 0), 1)
                    #cv2.arrowedLine(frame, (x, y), (x +5,y + 5), (255, 255, 0), 2)
                    cv2.rectangle(frame, (eye_cords[0][0] + face_coords[0] - 10 , eye_cords[0][1]  + face_coords[1] - 10), (eye_cords[0][2] + face_coords[0] + 10, eye_cords[0][3] + face_coords[1] + 10), (255, 255, 0), 2)
                    cv2.rectangle(frame, (eye_cords[1][0] + face_coords[0] - 10, eye_cords[1][1] + face_coords[1] - 10), (eye_cords[1][2] + face_coords[0] + 10, eye_cords[1][3] + face_coords[1] + 10), (255, 255, 0), 2)
                    cv2.rectangle(frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255, 255, 0), 2)
                    #cv2.imshow('prev', frame)

        out_video.write(frame)


        frame_count += 10
        cap.set(1, frame_count)

        if frame_count % 10 == 0:
            print(frame_count)

        if frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break

        if key_pressed == 27:
            break


    cap.release()
    cv2.destroyAllWindows()

    total_inference_time = time.time() - inference_time
    total_fps = frame_count / total_inference_time

    with open('result.txt', 'w') as f:
        f.write(str(total_loading_time) + '\n')
        f.write(str(total_inference_time) + '\n')
        f.write(str(total_fps) + '\n')














if __name__=='__main__':
    parser= ArgumentParser()
    parser.add_argument('--faceModel', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--input', default=None)
    parser.add_argument('--headpose', default=None)
    parser.add_argument('--facialLandmark', default=None)
    parser.add_argument('--gazeEstimation', default=None)
    parser.add_argument('--threshold', default=0.75)

    args=parser.parse_args()

    main(args)
