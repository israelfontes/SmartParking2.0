from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
from conf import Conf
from imutils.video import WebcamVideoStream
from imutils.video import VideoStream
from imutils.io import TempFile
from imutils.video import FPS
from datetime import datetime
from threading import Thread
import numpy as np 
import argparse
#import dropbox
import imutils
import dlib
import time
import cv2
import os

def upload_file(tempFile, client, imageID):
    print("[INFO] uploading {}...".format(imageID))
    path = "/{}.jpg".format(imageID)
    client.files_upload(open(tempFile.path, 'rb').read(), path)
    tempFile.cleanup()

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
    help="Path to the input configuration file")
args = vars(ap.parse_args())

conf = Conf(args["conf"]).load()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(conf['prototxt_path'], conf['model_path'])
print("[INFO] warming up camera...")

vs = VideoStream(0).start()
time.sleep(2.0)

H = None
W = None

ct = CentroidTracker(maxDisappeared=conf["max_disappear"])

trackers = []
trackableObjects = {}
totalFrames = 0
logFile = None
points=[("A","B"), ("B","C"),("C","D")]

fps = FPS().start()

while True:
    frame = vs.read()
    ts = datetime.now()
    newDate = ts.strftime("%m-%d-%y")

    if frame is None:
        break

    if logFile is None:
        logPath = os.path.join(conf['output_path'], conf['csv_name'])
        logFile = open(logPath, mode='a')

        pos = logFile.seek(0, os.SEEK_END)

        if conf['use_dropbox'] and pos == 0:
            logFile.write("Year,Month,Day,Time,Speed,ImageID\n")
        elif pos == 0:
            logFile.write("Year,Month,Day,Time,Speed")

    frame = imutils.resize(frame, width=conf["frame_width"])
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if W is None or H is None:
        (H,W) = frame.shape[:2]
        meterPerPixel = conf['distance']/W
    
    rects = []

    if totalFrames % conf['track_object'] == 0:
        trackers = []

        blob = cv2.dnn.blobFromImage(frame, size=(300,300), ddepth=cv2.CV_8U)
        net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5, 127.5, 127.5])
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0,0,i,2]

            if confidence > conf['confidence']:
                idx = int(detections[0,0,i,1])
            
                if CLASSES[idx] != "car":
                    continue

                box = detections[0,0,i,3:7] * np.array([W,H,W,H])
                (startX, startY, endX, endY) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                trackers.append(tracker)
    else:
        for tracker in trackers:
            tracker.update(rgb)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            rects.append((startX, startY, endX, endY))
    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)
        """
        if to is None:
            to = TrackableObject(objectID, centroid)
        elif not to.estimated:
            if to.direction is None:
                y = [c[0] for c in to.centroids]
                direction = centroid[0] - np.mean(y)
                to.direction = direction
            if to.direction > 0:
                if to.timestamp["A"] == 0:
                    if centroid[0] > conf["speed_estimation_zone"]["A"]:
                        to.timestamp["A"] = ts
                        to.position["A"] = centroid[0]
                elif to.timestamp["B"] == 0:
                    if centroid[0] > conf["speed_estimation_zone"]["B"]:
                        to.timestamp["B"] = ts
                        to.position["B"] = centroid[0]
                elif to.timestamp["C"] == 0:
                    if centroid[0] > conf["speed_estimation_zone"]["C"]:
                        to.timestamp["C"] = ts
                        to.position["C"] = centroid[0]
                elif to.timestamp["D"] == 0:
                    if centroid[0] > conf["speed_estimation_zone"]["D"]:
                        to.timestamp["D"] = ts
                        to.position["D"] = centroid[0]
                        to.lastPoint = True

            elif to.direction < 0:
                if to.timestamp["D"] == 0:
                    if centroid[0] > conf["speed_estimation_zone"]["D"]:
                        to.timestamp["D"] = ts
                        to.position["D"] = centroid[0]
                elif to.timestamp["C"] == 0:
                    if centroid[0] > conf["speed_estimation_zone"]["C"]:
                        to.timestamp["C"] = ts
                        to.position["C"] = centroid[0]
                elif to.timestamp["B"] == 0:
                    if centroid[0] > conf["speed_estimation_zone"]["B"]:
                        to.timestamp["B"] = ts
                        to.position["B"] = centroid[0]
                elif to.timestamp["A"] == 0:
                    if centroid[0] > conf["speed_estimation_zone"]["A"]:
                        to.timestamp["A"] = ts
                        to.position["A"] = centroid[0]
                        to.lastPoint = True
            if to.lastPoint and not to.estimated:
                estimatedSpeeds = []

                for(i,j) in points:
                    d = to.position[j] = to.position[i]
                    distanceInPixels = abs(d)

                    if distanceInPixels == 0:
                         continue
                    t = to.timestamp[j] - to.timestamp[i]
                    timeInSeconds = abs(t.total_seconds())
                    timeInHours = timeInSeconds / (60*60)

                    distanceInMeters = distanceInPixels * meterPerPixel
                    distanceInKm = distanceInMeters / 1000
                    estimatedSpeeds.append(distanceInKm/timeInHours)

                to.calculate_speed(estimatedSpeeds)

                to.estimated = True
                print("[INFO] Speed of the vehicle that just passed is: {:.2f}Km/h".format(to.speedKMH))
        """
        trackableObjects[objectID] = to
        
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10)
            , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4,
            (0, 255, 0), -1)
        """
        if not to.logged:
            # check if the object's speed has been estimated and it
            # is higher than the speed limit
            if to.estimated and to.speedMPH > conf["speed_limit"]:
                # set the current year, month, day, and time
                year = ts.strftime("%Y")
                month = ts.strftime("%m")
                day = ts.strftime("%d")
                time = ts.strftime("%H:%M:%S")
                # check if dropbox is to be used to store the vehicle
                # image
                if conf["use_dropbox"]:
                    # initialize the image id, and the temporary file
                    imageID = ts.strftime("%H%M%S%f")
                    tempFile = TempFile()
                    cv2.imwrite(tempFile.path, frame)
                    # create a thread to upload the file to dropbox
                    # and start it
                    t = Thread(target=upload_file, args=(tempFile,
                        client, imageID,))
                    t.start()
                    # log the event in the log file
                    info = "{},{},{},{},{},{}\n".format(year, month,
                        day, time, to.speedMPH, imageID)
                    logFile.write(info)
                # otherwise, we are not uploading vehicle images to
                # dropbox
                else:
                    # log the event in the log file
                    info = "{},{},{},{},{}\n".format(year, month,
                        day, time, to.speedMPH)
                    logFile.write(info)
                # set the object has logged
                to.logged = True
        """
    if conf["display"]:
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
            break
    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# check if the log file object exists, if it does, then close it
if logFile is not None:
    logFile.close()
# close any open windows
cv2.destroyAllWindows()
# clean up
print("[INFO] cleaning up...")
vs.stop()
