from utils_tracker.centroidtracker import CentroidTracker
from imutils.video import VideoStream, FileVideoStream
import numpy as np
import imutils
import cv2
import yaml
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from rethinkdb import RethinkDB
from datetime import datetime
import json
import socket
import time
from imutils.video import VideoStream
import imagezmq
from streaming import Streaming
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import os
from utils.plots import plot_one_box
import screenshot

sender = imagezmq.ImageSender("tcp://*:{}".format(3000), REQ_REP=False)

def pointInRect(point, rect):
    if point[0] > rect[0] and point[0] < rect[0]+rect[2] and point[1] > rect[1] and point[1] < rect[1]+rect[3]:
        return True
    return False

def sentido(vetor):
    a, b = 0, 0
    for i, p in enumerate(vetor):
        if i < len(vetor)-2 and vetor[i] > vetor[i+1]:
            a += 1
        elif i < len(vetor)-2 and vetor[i] < vetor[i+1]:
            b += 1
    if a > b:
        return 'saiu'
    return 'entrou'

def letterbox(img, new_shape=(512, 512), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

# construct the argument parse and parse the arguments
with open('config/config.json') as f:
  config = json.load(f)

# start rethink connection
r = RethinkDB()
r.connect(config['host_db'], config['port_db'], "SmartParking").repl()

if config['log']:
   print("RethinkDB Conect OK")
 
# start streaming
#stream = Streaming(config['port_stream'])

print("1 - draw points in image")
print("2 - start parking detection")

#option = input()
option = 2
if option == '1':
    screenshot.drawPoints(config['input_video'], config['yaml_parking'])

# Read YAML data (parking space polygons)
with open(config['yaml_parking'], 'r') as stream:
    parking_data = yaml.safe_load(stream)
parking_contours = []
parking_bounding_rects = []
parking_mask = []

for park in parking_data:
    points = np.array(park['points'])
    rect = cv2.boundingRect(points)
    points_shifted = points.copy()
    points_shifted[:,0] = points[:,0] - rect[0] # shift contour to roi
    points_shifted[:,1] = points[:,1] - rect[1]
    parking_contours.append(points)
    parking_bounding_rects.append(rect)
    mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                            color=255, thickness=-1, lineType=cv2.LINE_8)
    mask = mask==255
    parking_mask.append(mask)

# Loading model and config yolov5
device = select_device('')
half = device.type != 'cpu'
model = attempt_load(config['weights'], map_location=device)
imgsz = check_img_size(512, s=model.stride.max())  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


# Run inference
img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
_ = model(img.half() if half else img) if device.type != 'cpu' else None

if config['log']:
	print("[INFO] loading model...")
	print("[INFO] Weights file "+config['weights'])

#cap = cv2.VideoCapture(config['input_video'])
vs = VideoStream(src="rtsp://admin:teste123!@10.7.20.60:554/Streaming/Channels/101",usePicamera=False, resolution=(1000,600)).start()
#vs = VideoStream(src='http://131.95.3.162:80/mjpg/video.mjpg').start()
#print(vs.read())
time.sleep(2.0)
print(config['input_video'])
vid_path, vid_writer = None, None

if config['log']:
	print("[INFO] starting video stream...")
	print("[INFO] starting detection...")


tracker = CentroidTracker()

parking_status = [None]*len(parking_data)
parking_changed_count = [0]*len(parking_data)

while True:
    frame = vs.read()
    #print(frame)
    frame = imutils.resize(frame,width=1000, height=600)

    img = letterbox(frame, new_shape=512)[0]
    img0=frame
    rects = []

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    pred = model(img, augment=True)[0]
    pred = non_max_suppression(pred, config['confidence'], config['confidence'], classes=[2,3,5,6,7], agnostic=True)
    
    # Process detections
    for i, det in enumerate(pred):
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                rects.append((x1,y1,x2,y2))
         

    objects = tracker.update(rects)
    
    parking_status_buffer = [None]*len(parking_data)

    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = 'CAR'+str(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        for index, park in enumerate(parking_data):
            points = np.array(park['points'])
            rect = cv2.boundingRect(points)
            if pointInRect(centroid, rect):
                parking_status_buffer[index] = text
    
    change = False

    if config['log'] and parking_status != parking_status_buffer:
        for i in range(0,len(parking_status_buffer)):
            if parking_status[i] != parking_status_buffer[i]:
                if parking_changed_count[i] == config['park_wait']:
                    date_now = datetime.now()
                    date = date_now.strftime('%d/%m/%Y-%H:%M:%S')
                    parking_changed_count[i] = 0
                    change = True
                else: parking_changed_count[i] += 1 
        if change: 
            parking_status = parking_status_buffer
            r.db('SmartParking').table('parking').insert([{
                        "parking":parking_status,
                        "occupied": parking_status_buffer[i],
                        "date": date
                    }]).run()

        print(parking_status)
        print('[INFO] alteration in parking')

    if config['parking_overlay']:
        for index, park in enumerate(parking_data):
            points = np.array(park['points'])
            if parking_status[index] != None: color = (0,0,255)
            else: color = (0,255,0)
            cv2.drawContours(frame, [points], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_8)
            moments = cv2.moments(points)        
            centroid = (int(moments['m10']/moments['m00'])-3, int(moments['m01']/moments['m00'])+3)
            cv2.putText(frame, str(park['id']), (centroid[0]+1, centroid[1]+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, str(park['id']), (centroid[0]-1, centroid[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, str(park['id']), (centroid[0]+1, centroid[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, str(park['id']), (centroid[0]-1, centroid[1]+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, str(park['id']), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

    if config['save_video'] and vid_path != config['save_video_path']:  # new video
        vid_path = config['save_video_path']
        if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()  # release previous video writer
        if config['input_cam_or_file']:  # video
            fps = 5
            w = frame.shape[1]
            h = frame.shape[0]
        else:  # stream
            fps, w, h = 5, frame.shape[1], frame.shape[0]
        vid_writer = cv2.VideoWriter(config['save_video_path'], cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if config['save_video']:
        vid_writer.write(frame)

    # show the output frame
    if config['show_output']:
        cv2.imshow("Frame", frame)
    
    # streaming
    if config['stream_result']:
        ret_code, jpg_buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        sender.send_jpg("10.7.49.166", jpg_buffer)

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
if config['show_output']:
    cv2.destroyAllWindows()
