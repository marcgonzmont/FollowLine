#!/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
import cv2
import threading
import time
from datetime import datetime

time_cycle = 80

class MyAlgorithm(threading.Thread):

    def __init__(self, cameraL, cameraR, motors):
        self.cameraL = cameraL
        self.cameraR = cameraR
        self.motors = motors
        self.imageRight=None
        self.imageLeft=None
        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        threading.Thread.__init__(self, args=self.stop_event)

    def setRightImageFiltered(self, image):
        self.lock.acquire()
        self.imageRight=image
        self.lock.release()


    def setLeftImageFiltered(self, image):
        self.lock.acquire()
        self.imageLeft=image
        self.lock.release()

    def getRightImageFiltered(self):
        self.lock.acquire()
        tempImage=self.imageRight
        self.lock.release()
        return tempImage

    def getLeftImageFiltered(self):
        self.lock.acquire()
        tempImage=self.imageLeft
        self.lock.release()
        return tempImage

    def run (self):

        while (not self.kill_event.is_set()):
            start_time = datetime.now()
            if not self.stop_event.is_set():
                self.execute()
            finish_Time = datetime.now()
            dt = finish_Time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            #print (ms)
            if (ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)

    def stop (self):
        self.stop_event.set()

    def play (self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill (self):
        self.kill_event.set()

    def execute(self):
        #GETTING THE IMAGES
        imageLeft = self.cameraL.getImage().data
        # imageRight = self.cameraR.getImage().data

        imgLeft, img_L_filtered = getPoints(imageLeft)

        # Add your code here
        # print ("Running")

        #EXAMPLE OF HOW TO SEND INFORMATION TO THE ROBOT ACTUATORS
        #self.motors.sendV(10)
        #self.motors.sendW(5)

        #SHOW THE FILTERED IMAGE ON THE GUI
        self.setRightImageFiltered(np.uint8(img_L_filtered))
        self.setLeftImageFiltered(np.uint8(imageLeft))

def getPoints(img):
        kernel_b = (5,5)
        kernel_c = (3,3)

        lower_color = np.array([0, 100, 100])
        upper_color = np.array([180, 255, 255])

        blur = cv2.GaussianBlur(img, kernel_b, 0)
        img_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, lower_color, upper_color)
        mask = cv2.erode(mask, kernel= kernel_c, iterations= 2)
        mask = cv2.dilate(mask, kernel= kernel_c, iterations= 2)

        h, w = mask.shape[:2]
        h_m = h // 2
        # ref_points = {}
        # curr_points = {}

        for y in range(h_m, h, h_m // 5):
            cv2.line(img, (0, y), (w, y), (255, 0, 0), 1)
            whites = np.nonzero(mask[y,:])
            # x = whites[0]+whites[len(whites[0]) - 1]/2
            print(x)
            # cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), 3)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        # test = np.hstack([img, mask])

        return img, mask
