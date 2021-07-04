# -*- coding: utf-8 -*-

import yaml
import numpy as np
import cv2
from imutils.video import VideoStream
import imutils

def screenshotVideo(input_video):
    vs = VideoStream(input_video).start()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame,width=1000, height=600)
        cv2.imshow('frame', frame)
       
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('c'):
            cv2.imwrite('../data/frame.jpg', frame)
            break
        elif k == ord('j'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_cur_frame+1000) # jump to frame

    cv2.destroyAllWindows()    
    pass

refPt = []
cropping = False

def drawPoints(input_video, file_yml):
    
    data = []
    
    screenshotVideo(input_video)
    
    img = cv2.imread('../data/frame.jpg')

    def yaml_loader(file_yml):
        with open(file_yml, "r") as file_descr:
            data = yaml.load(file_descr)
            return data

    def yaml_dump(file_yml, data):
        with open(file_yml, "a") as file_descr:
            yaml.dump(data, file_descr)


    def yaml_dump_write(file_yml, data):
        with open(file_yml, "w") as file_descr:
            yaml.dump(data, file_descr)

    def click_and_crop(event, x, y, flags, param):
        current_pt = {'id': 0, 'points': []}
        # grab references to the global variables

        global refPt, cropping
        if event == cv2.EVENT_LBUTTONDOWN:
            print('envent')
            refPt.append((x, y))
            cropping = False
        if len(refPt) == 4:
            if data == []:
                if yaml_loader(file_yml) != None:
                    data_already = len(yaml_loader(file_yml))
                else:
                    data_already = 0
            else:
                if yaml_loader(file_yml) != None:
                    data_already = len(data) + len(yaml_loader(file_yml))
                else:
                    data_already = len(data) 
            
            cv2.line(image, refPt[0], refPt[1], (0, 255, 0), 1)
            cv2.line(image, refPt[1], refPt[2], (0, 255, 0), 1)
            cv2.line(image, refPt[2], refPt[3], (0, 255, 0), 1)
            cv2.line(image, refPt[3], refPt[0], (0, 255, 0), 1)

            temp_lst1 = list(refPt[2])
            temp_lst2 = list(refPt[3])
            temp_lst3 = list(refPt[0])
            temp_lst4 = list(refPt[1])

            current_pt['points'] = [temp_lst1, temp_lst2, temp_lst3, temp_lst4]
            current_pt['id'] = data_already
            data.append(current_pt)
            # data_already+=1
            refPt = []
    
    image = img
    
    cv2.namedWindow("Double click to mark points")
    cv2.imshow("Double click to mark points", image)
    cv2.setMouseCallback("Double click to mark points", click_and_crop)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("Double click to mark points", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
           
    # data list into yaml file
    if data != []:
        yaml_dump(file_yml, data)
    cv2.destroyAllWindows() #important to prevent window from becoming inresponsive
    pass