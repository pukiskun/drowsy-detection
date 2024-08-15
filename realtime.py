import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import winsound

class VideoStream:
    """Camera object that controls video streaming from the webcam in a separate processing thread."""
    def __init__(self, resolution=(640, 480), framerate=30, camera_index=0):
        self.stream = cv2.VideoCapture(camera_index)
        if not self.stream.isOpened():
            raise ValueError("Unable to open webcam. Make sure it is connected properly.")
        
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5, type=float)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH', default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')
    return parser.parse_args()

def switch_camera(videostream, camera_index):
    videostream.stop()
    time.sleep(1)
    return VideoStream(resolution=(imW, imH), framerate=30, camera_index=camera_index).start()

def main():
    args = parse_args()

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = args.threshold
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu

    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    if use_TPU and GRAPH_NAME == 'detect.tflite':
        GRAPH_NAME = 'edgetpu.tflite'

    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

    labels = load_labels(PATH_TO_LABELS)
    if labels[0] == '???':
        del(labels[0])

    try:
        if use_TPU:
            interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        else:
            interpreter = Interpreter(model_path=PATH_TO_CKPT)
        interpreter.allocate_tensors()
    except Exception as e:
        sys.exit(f"Error loading model: {e}")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    outname = output_details[0]['name']
    if 'StatefulPartitionedCall' in outname:
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    current_camera = 0
    try:
        videostream = VideoStream(resolution=(imW, imH), framerate=30, camera_index=current_camera).start()
        time.sleep(1)
    except Exception as e:
        sys.exit(f"Error initializing video stream: {e}")

    mengantuk_start_time = None
    mengantuk_count = 0

    tertidur_start_time = None
    tertidur_count = 0

    def nothing(x):
        pass

    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 500, 100)
    cv2.createTrackbar('Mengantuk Threshold', 'Controls', 3, 10, nothing)
    cv2.createTrackbar('Tertidur Threshold', 'Controls', 3, 10, nothing)

    while True:
        mengantuk_threshold = cv2.getTrackbarPos('Mengantuk Threshold', 'Controls')
        tertidur_threshold = cv2.getTrackbarPos('Tertidur Threshold', 'Controls')

        t1 = cv2.getTickCount()
        frame1 = videostream.read()
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        mengantuk_detected = False
        tertidur_detected = False

        for i in range(len(scores)):
            if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                object_name = labels[int(classes[i])]
                label = f'{object_name}: {int(scores[i] * 100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                if object_name == "supir-mengantuk":
                    current_time = time.time()
                    if mengantuk_start_time is None:
                        mengantuk_start_time = current_time
                    elif current_time - mengantuk_start_time >= 1:
                        mengantuk_count += 1
                        mengantuk_start_time = current_time
                    if mengantuk_count >= mengantuk_threshold:
                        winsound.Beep(800, 500)  # Frequency 800 Hz, duration 500 ms
                        mengantuk_count = 0  # Reset counter after beeping
                        mengantuk_detected = True
                else:
                    mengantuk_start_time = None  # Reset start time if other object detected

                if object_name == "supir-tertidur":
                    current_time = time.time()
                    if tertidur_start_time is None:
                        tertidur_start_time = current_time
                    elif current_time - tertidur_start_time >= 1:
                        tertidur_count += 1
                        tertidur_start_time = current_time
                    if tertidur_count >= tertidur_threshold:
                        winsound.Beep(1200, 500)  # Frequency 1200 Hz, duration 500 ms
                        tertidur_count = 0  # Reset counter after beeping
                        tertidur_detected = True
                else:
                    tertidur_start_time = None  # Reset start time if other object detected

        cv2.putText(frame, f'Mengantuk Count: {mengantuk_count}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Tertidur Count: {tertidur_count}', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Mengantuk', frame)

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            current_camera = (current_camera + 1) % 10  # Cycle through 10 possible camera indices
            videostream = switch_camera(videostream, current_camera)

    cv2.destroyAllWindows()
    videostream.stop()

if __name__ == '__main__':
    main()
