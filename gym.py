from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import os
import argparse
from necro import Necro
from Utils import Utils


class Gym:

    def __init__(self):
        # TODO move the common constants to a separate file
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
        self.a2 = self.parser.add_argument('--ignoredPoints', '--list', nargs='+', required=False,
                                           help="Enter undesirable body parts from list below" + "\n*************" +
                                                self.pose_pairs,
                                           default=[])
        self.a3 = self.parser.add_argument('--epochs', required=False, default=10, help='number of epochs')
        self.a4 = int(self.parser.add_argument('--necro', required=False, default=0, help='run necro (0 | 1)'))
        self.a5 = int(self.parser.add_argument('--training', required=False, default=1, help='run training (0 | 1)'))
        self.args = self.parser.parse_args()

        if len(self.args.ignoredPoints) % 2 != 0:
            raise Exception("Each body part should contain two body points")

        temp_pairs = self.POSE_PAIRS.copy()
        for i in range(0, len(self.args.ignoredPoints), 2):
            i = self.args.ignoredPoints[i:i + 2]
            if i not in self.POSE_PAIRS:
                raise Exception("Entered wrong body part.Use help command for list of body parts.")
            else:
                temp_pairs.remove(i)

        self.train_batch(self.args.inputPath, self.args.ignoredPoints, self.args.epochs)

    @staticmethod
    def train_exercise(name, training_data_path, epochs):
        training_data = os.path.dirname(training_data_path) + "/processed_images/" + name

        number_of_classes = Utils.get_folder_list_in_folder(training_data_path, True)

        # CNN model
        classifier = Sequential()

        classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(32, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(units=128, activation='relu'))
        classifier.add(Dense(units=number_of_classes, activation='softmax'))  # number of classes

        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Data Augmentation
        batch_size = 32
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.3,
                                           zoom_range=0.4,
                                           rotation_range=10,
                                           horizontal_flip=True)

        training_generator = train_datagen.flow_from_directory(
            training_data,
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training')

        test_generator = train_datagen.flow_from_directory(
            training_data,
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='categorical',
            subset='test')

        classifier.fit(training_generator,
                       steps_per_epoch=training_generator.samples // batch_size,
                       epochs=epochs,
                       validation_data=test_generator,
                       validation_steps=test_generator.samples // batch_size)

        classifier.save(name + '.h5')

    def train_batch(self, directory, ignored_points, epochs):
        epochs = int(epochs)
        if int(self.args.necro):
            print('running necro')
            Necro.run_analyze(directory, ignored_points)

        if int(self.args.training):
            exercise_paths = Utils.get_folder_list_in_folder(directory, False)
            for exercise_path in exercise_paths:
                if not exercise_path.endswith('processed_images'):
                    split = exercise_path.split('\\')
                    name = split[len(split) - 1]
                    print(name + ' training started. ****')
                    self.train_exercise(name, exercise_path, epochs)
                    print(name + ' training finished. ****')

        # print('removing empty folders.')
        # Utils.remove_empty_folders(directory)
        # print('Empty folders removed.')

# a = Gym()
