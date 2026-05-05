class PredictionTracker:
    def __init__(self, none_class=None, threshold=0.6):
        self.none_class = none_class
        self.threshold = threshold
        self.detection_count = {}
        self.person_predictions = {}

    def update(self, obj_id, pred, confidence, label_name):
        if (self.none_class is None and confidence > self.threshold) or \
           (self.none_class is not None and pred != self.none_class and confidence > self.threshold):

            self.detection_count[obj_id] = self.detection_count.get(obj_id, 0) + 1
        else:
            self.detection_count[obj_id] = 0

        if self.detection_count[obj_id] >= 2:
            self.person_predictions[obj_id] = (label_name.upper(), confidence)
        else:
            self.person_predictions[obj_id] = ("NONE", confidence)

        return self.person_predictions[obj_id]