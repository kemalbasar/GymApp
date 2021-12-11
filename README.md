<h1 align="center">Pose Estimation with Tensorflow</h1>
<p align="center">Pose Estimation using PoseNet model with Tensorflow and Python</p>



## Files
There are four files in the directory:

1. **posenet_mobilenet.tflite** - The actual Tensorflow Lite model;
2. **PhotoDemo.py** - The code of analizing picture
3. **CamDemo.py** - The code of analizing live , based on PhotoDemo.py
4. **OutputWriter.py** - The file which writes outsputs of pictures to Json file.


## How to use

1.import Analize_Posture from OutputWriter.py.

2.Your only input will be the variable which storing path where  pictures you want to analize stands.

3.Script will analize all pictures in path and write it to Json file which name is cordinates.json.
   to the same path
   

