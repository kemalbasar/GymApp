from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import os
from Utils import Utils


def train_exercise(name, training_data_path, epochs):
    number_of_classes = Utils.get_folder_list_in_folder(training_data_path, True)

    print(number_of_classes)
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
        training_data_path,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    test_generator = train_datagen.flow_from_directory(
        training_data_path,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    classifier.fit(training_generator,
                   steps_per_epoch=training_generator.samples // batch_size,
                   epochs=epochs,
                   validation_data=test_generator,
                   validation_steps=test_generator.samples // batch_size)

    classifier.save(name + '.h5')


train_exercise('squat', 'C:/Users/mertd/Desktop/acf-video-analysis/training_data/image_data/processed_images/squat', 10)
