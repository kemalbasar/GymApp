# Generic imports
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# global variables

BODY_PARTS = {"nose": 0, "leftEye": 1, "rightEye": 2, "leftEar": 3, "rightEar": 4,
              "leftShoulder": 5, "rightShoulder": 6, "leftElbow": 7, "rightElbow": 8, "leftWrist": 9,
              "rightWrist": 10, "leftHip": 11, "rightHip": 12, "leftKnee": 13, "rightKnee": 14,
              "leftAnkle": 15, "rightAnkle": 16}

BODY_PARTS_COLORS = {"nose": (153, 0, 0), "leftEye": (153, 0, 153), "rightEye": (102, 0, 153),
                     "leftEar": (153, 0, 50),
                     "rightEar": (153, 0, 102),
                     "leftShoulder": (51, 153, 0), "rightShoulder": (153, 102, 0), "leftElbow": (0, 153, 0),
                     "rightElbow": (153, 153, 0), "leftWrist": (0, 153, 51),
                     "rightWrist": (102, 153, 0), "leftHip": (0, 51, 153), "rightHip": (0, 153, 102),
                     "leftKnee": (0, 0, 153), "rightKnee": (0, 153, 153),
                     "leftAnkle": (51, 0, 153), "rightAnkle": (0, 102, 153)}

POSE_PAIRS = [['nose', 'leftEye'], ['nose', 'rightEye'], ['rightEye', 'rightEar'],
              ['leftEye', 'leftEar'], ['rightShoulder', 'rightElbow'], ['leftShoulder', 'leftElbow'],
              ['rightElbow', 'rightWrist'], ['leftElbow', 'leftWrist'], ['rightShoulder', 'leftShoulder'],
              ['rightShoulder', 'rightHip'], ['leftShoulder', 'leftHip'], ['rightHip', 'rightKnee'],
              ['leftHip', 'leftKnee'], ['rightKnee', 'rightAnkle'], ['leftKnee', 'leftAnkle'],
              ['rightHip', 'leftHip']]

# interpreter = tf.lite.Interpreter('/home/basar/Documents/GitHub/machine-learning/Assets/posenet_mobilenet.tflite')
interpreter = tf.lite.Interpreter('C:/Users/mertd/Desktop/acf-video-analysis/machine-learning/Assets/posenet_mobilenet.tflite')

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

inHeight = input_details[0]['shape'][1]
inWidth = input_details[0]['shape'][2]


class LivePrediction:

    def __init__(self, model_name):

        self.classifier = load_model(model_name)

    @staticmethod
    def analyze_frame(frame):
        undefined_points = 0

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        in_frame = cv2.resize(frame, (inWidth, inHeight))
        in_frame = cv2.cvtColor(in_frame, cv2.COLOR_BGR2RGB)

        in_frame = np.expand_dims(in_frame, axis=0)
        in_frame = (np.float32(in_frame) - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], in_frame)

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        offset_data = interpreter.get_tensor(output_details[1]['index'])

        heatmaps = np.squeeze(output_data)
        offsets = np.squeeze(offset_data)

        points = []
        conf_list = []
        joint_num = heatmaps.shape[-1]

        for i in range(heatmaps.shape[-1]):
            joint_heatmap = heatmaps[..., i]
            max_val_pos = np.squeeze(np.argwhere(joint_heatmap == np.max(joint_heatmap)))
            remap_pos = np.array(max_val_pos / 8 * 257, dtype=np.int32)
            point_y = int(remap_pos[0] + offsets[max_val_pos[0], max_val_pos[1], i])
            point_x = int(remap_pos[1] + offsets[max_val_pos[0], max_val_pos[1], i + joint_num])
            conf = np.max(joint_heatmap)

            x = (frame_width * point_x) / inWidth
            y = (frame_height * point_y) / inHeight

            if conf > 0:
                points.append((int(x), int(y)))

            else:
                points.append(None)
                undefined_points = undefined_points + 1

            conf_list.append(conf)

        return points, undefined_points, frame_height, frame_width

    @staticmethod
    def draw_lines(frame, points, undefined_points):

        if undefined_points / 17 < 7 / 10:

            for pair in POSE_PAIRS:

                part_from = pair[0]
                part_to = pair[1]

                id_from = BODY_PARTS[part_from]
                id_to = BODY_PARTS[part_to]

                color_from = BODY_PARTS_COLORS[part_from]
                color_to = BODY_PARTS_COLORS[part_to]

                if points[id_from] and points[id_to]:
                    cv2.line(frame, points[id_from], points[id_to], color_from, 3)
                    cv2.ellipse(frame, points[id_from], (3, 3), 0, 0, 360, color_from, cv2.FILLED)
                    cv2.ellipse(frame, points[id_to], (3, 3), 0, 0, 360, color_to, cv2.FILLED)

        return frame

    def predict(self, frame):

        points, undefined_points, frame_height, frame_width = LivePrediction.analyze_frame(frame)
        img = LivePrediction.create_empty_picture(frame_height, frame_width)
        frame = LivePrediction.draw_lines(img, points, undefined_points)
        test_image = cv2.resize(img, (64, 64))
        test_image = test_image[..., ::-1]
        test_image = np.expand_dims(test_image, axis=0)
        result = self.classifier.predict(test_image)
        prediction = "None"
        if result[0][0] == 1:
            prediction = 'down'  # predictions in array are in alphabetical order
        elif result[0][1] == 1:
            prediction = 'up'

        return prediction, points, undefined_points, frame

    @staticmethod
    def create_empty_picture(frame_height, frame_width):

        img = np.zeros((frame_height, frame_width, 3), np.uint8)

        return img
