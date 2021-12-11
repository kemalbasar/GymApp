import cv2
import streamlit as st
from live_prediction_demo import LivePrediction
from rep_counter import Counter

col1, col2 = st.beta_columns(2)

col1.header("Upload your model and start analyze.")
prediction_text = col1.empty()
counter_text = col1.empty()
skeleton_window = col1.image([])
analyze_session = col1.checkbox("analyze_session")

model_choice = st.radio("Which model you want to test", ("squat", "pushup", "shouldertap"))

if model_choice == "squat":
    model = "C:/Users/mertd/Desktop/acf-video-analysis/machine-learning/squat.h5"
elif model_choice == "pushup":
    model = "/home/basar/Documents/GitHub/machine-learning/pushup.h5"
elif model_choice == "shouldertap ":
    model = "/home/basar/Documents/GitHub/machine-learning/shouldertap.h5"


col2.header("Live Cam")
body_window = col2.image([])
s = col2.empty()


picture_name = col2.text_input("Give a name for your picture", "data_to_train")
button = col2.button("take shot")


def runner():
    session = LivePrediction(model)
    counter_instance = Counter()
    cap = cv2.VideoCapture(0)
    while analyze_session:
        _, frame = cap.read()
        body_window.image(frame)
        if button:
            record_data(frame)
        prediction, points, undefined_points, frame_skeleton = session.predict(frame)
        counter_instance.countExercise(prediction)
        skeleton_window.image(frame_skeleton)
        prediction_text.markdown(prediction * 5)
        counter_text.markdown(counter_instance.exerciseCounter)

    cap.release()
    cv2.destroyAllWindows()


def record_data(frame):
    cv2.imwrite("/home/basar/Documents/GitHub/machine-learning/Data to Train/" + picture_name + ".jpeg", frame)


runner()
