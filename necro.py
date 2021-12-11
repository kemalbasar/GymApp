import json
import os
import shutil
from src.PhotoDemo import draw_skeleton, BODY_PARTS
import numpy as np
import argparse


class Necro:

    def __init__(self):
        self.pose_pairs = "[['nose','leftEye'], ['nose','rightEye'], ['rightEye','rightEar']," \
                          " ['leftEye','leftEar'],['rightShoulder','rightElbow'], ['leftShoulder','leftElbow']," \
                          " ['rightElbow','rightWrist'],['leftElbow','leftWrist'], ['rightShoulder','leftShoulder']," \
                          " ['rightShoulder','rightHip'],['leftShoulder','leftHip'], ['rightHip','rightKnee']," \
                          " ['leftHip','leftKnee'], ['rightKnee','rightAnkle'],['leftKnee','leftAnkle']," \
                          "['rightHip','leftHip']]"

        self.POSE_PAIRS = [['nose', 'leftEye'], ['nose', 'rightEye'], ['rightEye', 'rightEar'],
                           ['leftEye', 'leftEar'], ['rightShoulder', 'rightElbow'], ['leftShoulder', 'leftElbow'],
                           ['rightElbow', 'rightWrist'], ['leftElbow', 'leftWrist'], ['rightShoulder', 'leftShoulder'],
                           ['rightShoulder', 'rightHip'], ['leftShoulder', 'leftHip'], ['rightHip', 'rightKnee'],
                           ['leftHip', 'leftKnee'], ['rightKnee', 'rightAnkle'], ['leftKnee', 'leftAnkle'],
                           ['rightHip', 'leftHip']]

        self.parser = argparse.ArgumentParser()
        self.a1 = self.parser.add_argument('--inputPath', required=True, help='path to image folder')
        self.a2 = self.parser.add_argument('--ignoredPoints', '--list', required=False, nargs='+',
                                           help="Enter undesirable body parts from list below" + "\n*************" +
                                                self.pose_pairs,
                                           default=[])
        self.args = self.parser.parse_args()

        if len(self.args.ignoredPoints) % 2 != 0:
            raise Exception("Each body part should contain two body points")

        self.needed_body_parts = self.POSE_PAIRS.copy()
        for i in range(0, len(self.args.ignoredPoints), 2):
            a = self.args.ignoredPoints[i:i + 2]
            if a not in self.POSE_PAIRS:
                raise Exception("Entered wrong body part.Use help command for list of body parts.")
            else:
                self.needed_body_parts.remove(a)

        self.directory = self.args.inputPath
        self.ignored_points = self.args.ignoredPoints
        self.length_of_ignored_points = len(set(self.ignored_points))

        self.coordinate_dict = {}
        self.name_list_of_bparts = [*BODY_PARTS]

        self.run_analyze(self.directory, self.ignored_points)

    def run_analyze(self, dir_path, ignored_points):

        if os.path.exists(dir_path):

            os.makedirs(dir_path + "/processed_images")
            subdirectories = os.listdir(dir_path)
            subdirectories.remove("processed_images")

            for subdirectory in subdirectories:
                os.makedirs(dir_path + "/processed_images/" + os.path.basename(subdirectory))

                for input_directory in os.listdir(dir_path + "/" + subdirectory):
                    output_directory = dir_path + "/processed_images/" + os.path.basename(subdirectory) + "/" + \
                                       input_directory
                    os.makedirs(output_directory)
                    self.analyse_pictures(dir_path + "/" + os.path.basename(subdirectory) + "/" + input_directory,
                                          output_directory)

            print("skeletons Up!!")

        else:
            print("path does not exist!!")

    def analyse_pictures(self, input_directory, output_directory):

        for picture_name in os.listdir(input_directory):
            if picture_name.endswith(('jpg', 'png')):
                output = draw_skeleton(input_directory + "/" + picture_name, output_directory,
                                       self.needed_body_parts, self.length_of_ignored_points)
        #               Necro.create_json(picture_name, output)


# a = Necro()
