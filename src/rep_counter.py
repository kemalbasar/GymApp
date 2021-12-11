validFrameThreshold = 3
validThreshold = 0.93
invalidThreshold = 0.90


class Counter:

    def __init__(self):

        self.exerciseDownState = 0
        self.exerciseUpState = 0
        self.exerciseCompleteState = 0
        self.exerciseCounter = 0

    def resetStates(self):

        self.exerciseUpState = 0
        self.exerciseDownState = 0
        self.exerciseCompleteState = 0

    def countExercise(self, prediction):

        # var confidence = prediction.probability.toFixed(3)
        #   frameIsValid = !prediction.className.includes('partial') && confidence > invalidThreshold
        #   if (frameIsValid)

        if prediction == "up":  # && confidence > validThreshold

            if self.exerciseUpState > validFrameThreshold and self.exerciseDownState > validFrameThreshold:

                self.exerciseCompleteState = self.exerciseCompleteState + 1

                if self.exerciseCompleteState > validFrameThreshold:
                    self.exerciseCounter = self.exerciseCounter + 1
                    self.resetStates()

            else:

                self.exerciseUpState = self.exerciseUpState + 1

        elif prediction == "down" and self.exerciseUpState > validFrameThreshold:  # &&confidence > validThreshold

            self.exerciseDownState = self.exerciseDownState + 1

    #    displayValid(frameIsValid)
    #    validFrameBufferLoad(frameIsValid)
