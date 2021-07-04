from utils_tracker.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

ct = CentroidTracker()

(H,W) = (None, None)

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("[INFO] CaffeModel file"+args['model'])
print("[INFO] Proto file"+args['prototxt'])
print("[INFO] starting video stream...")

rtsp_url = 'http://131.95.3.162:80/mjpg/video.mjpg'#'rtsp://admin:teste123!@10.7.20.60:554/Streaming/Channels/101'
video = '../data/1.mp4'
vs = VideoStream(rtsp_url).start()
time.sleep(2.0)

CLASSES = ["background", "person", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor", "truck"]

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    if W is None or H is None:
        (H,W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, size=(400,400), ddepth=cv2.CV_8U)
    net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5, 127.5, 127.5])
    detections = net.forward()
    rects = []

    for i in range(0, detections.shape[2]):
        if detections[0,0,i,2] > args["confidence"]:
            
            #idx = int(detections[0,0,i,1])
            
            #if CLASSES[idx] != "car":
            #    continue
            
            #objectID = 'carID: {}'.format(i)

            box = detections[0,0,i,3:7] * np.array([W,H,W,H])
            #rects.append((box.astype("int"), objectID))
            rects.append(box.astype("int"))
            #[(box,id), (box,id), (box,id)]

            #print("Confidence: {}".format(detections[0,0,i,2]))

            (startX,startY,endX,endY) = box.astype("int")
            cv2.rectangle(frame,(startX, startY), (endX, endY), (0,255,0),2)

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()