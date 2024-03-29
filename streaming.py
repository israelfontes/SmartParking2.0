"""pub_sub_broadcast.py -- broadcast OpenCV stream using PUB SUB."""

import sys

import socket
import traceback
from time import sleep
import cv2
from imutils.video import VideoStream
import imagezmq

class Streaming():
    def __init__(self,port=5555):
        
        sender = imagezmq.ImageSender("tcp://*:{}".format(port), REQ_REP=False)
        print(port)
        # Open input stream; comment out one of these capture = VideoStream() lines!
        # *** You must use only one of Webcam OR PiCamera
        # Webcam source for broadcast images
        # capture = VideoStream()  # Webcam
        # PiCamera source for broadcast images (Raspberry Pi only)
        # capture = VideoStream(usePiCamera=True)  # PiCamera

        #capture.start()
        #sleep(2.0)  # Warmup time; needed by PiCamera on some RPi's
        print("Input stream opened")

        # JPEG quality, 0 - 100
        jpeg_quality = 95
        # Send RPi hostname with each image
        # This might be unnecessary in this pub sub mode, as the receiver will
        #    already need to know our address and can therefore distinguish streams
        # Keeping it anyway in case you wanna send a meaningful tag or something
        #    (or have a many to many setup)
        rpi_name = socket.gethostname()
        print("rpi_name")
    def send(self, frame):
        print("send1") 
        ret_code, jpg_buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        self.sender.send_jpg("10.7.49.166", jpg_buffer)
        print("send")
        #except (KeyboardInterrupt, SystemExit):
        #    print('Exit due to keyboard interrupt')
        #except Exception as ex:
        #    print('Python error with no Exception handler:')
        #    print('Traceback error:', ex)
        #    traceback.print_exc()
        #finally:
        #    self.capture.stop()
        #    self.sender.close()
        #    sys.exit()
