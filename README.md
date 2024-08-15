# Drowsy Detection

Drowsy Detection is a real-time application that uses a TensorFlow Lite model to detect drowsiness in drivers via a webcam. The application alerts the user with sound notifications if drowsiness or sleep is detected.

## Features
1. Real-time drowsiness detection using TensorFlow Lite models.
2. Adjustable thresholds for drowsiness and sleep detection.
3. Supports Coral Edge TPU Accelerator for faster inference (optional).
4. Cycle through multiple camera indices.
5. Visual display of detected objects and counts.
---
## Requirements
1. Python 3.x
2. OpenCV
3. TensorFlow Lite
4. (Optional) Coral Edge TPU library

## Installation
1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/drowsy-detection.git
    cd drowsy-detection
    ```
2. **Install dependencies:**

    ```bash
    pip install opencv-python tflite-runtime numpy
    ```

3. **(Optional) Install Coral Edge TPU library:**

    Follow the [Coral documentation](https://coral.ai/docs/accelerator/get-started/) for installation instructions if you plan to use the Edge TPU.

## Usage

1. **Prepare your model and labels:**

   Place your TensorFlow Lite model (.tflite) and label map file (labelmap.txt) in a directory.

2. **Run the application:**

    ```bash
    python drowsy_detection.py --modeldir /path/to/modeldir --graph detect.tflite --labels labelmap.txt --threshold 0.5 --resolution 1280x720
    ```

    - `--modeldir`: Folder where the .tflite file is located.
    - `--graph`: Name of the .tflite file (default: detect.tflite).
    - `--labels`: Name of the label map file (default: labelmap.txt).
    - `--threshold`: Minimum confidence threshold for displaying detected objects (default: 0.5).
    - `--resolution`: Desired webcam resolution in WxH (default: 1280x720).
    - `--edgetpu`: Use Coral Edge TPU Accelerator (optional).

3. **Control the application:**

    - Press `q` to quit.
    - Press `c` to switch cameras (up to 10 indices).

## Configuration

- **Mengantuk Threshold**: The number of consecutive detections required to trigger a drowsiness alert.
- **Tertidur Threshold**: The number of consecutive detections required to trigger a sleep alert.
