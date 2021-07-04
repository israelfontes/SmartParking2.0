import numpy as np

class TrackableObject:
    def __init__(self, objectID, centroid):

        self.objectID = objectID
        self.centroids = [centroid]

        self.timestamp = {"A":0, "B":0, "C":0, "D":0}
        self.position = {"A":None, "B":None, "C": None, "D":None}
        self.lastPoint = False

        self.speedKMPH = None

        self.estimated = False
        self.logged = False

        self.direction = None
    
    def calculate_speed(self, estimatedSpeeds):
        self.speedKMPH = np.average(estimatedSpeeds)
    