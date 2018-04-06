#!/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
import cv2
import threading
import time
from datetime import datetime
from scipy.spatial import distance as dist

time_cycle = 80

class MyAlgorithm(threading.Thread):

    def __init__(self, cameraL, cameraR, motors):
        self.cameraL = cameraL
        self.cameraR = cameraR
        self.motors = motors
        self.imageRight=None
        self.imageLeft=None
        self.ref_points = None
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
        first = True
        # ref_points = None
        # curr_points = None

        while (not self.kill_event.is_set()):
            start_time = datetime.now()
            if not self.stop_event.is_set():
                self.execute(first)
                first = False
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

    def execute(self, first):
        #GETTING THE IMAGES
        imageLeft = self.cameraL.getImage().data
        divisions = 5
        # List of middle points of the road
        curr_points = None
        # Compute the frame and get the middle points of the road
        imgLeft, img_L_filtered, points = self.getPoints(imageLeft, divisions)
        # if len(points) <= 1:
        #     print("I CAN'T SEE THE ROAD!!")
        # else:
        #     print("OK!")
        # Set the reference points
        if first:
            self.ref_points = points

        # Other frames, get current points and compute distances with the reference points
        else:
            curr_points = points


        # distances = getDistances(ref_points, curr_points)

        #EXAMPLE OF HOW TO SEND INFORMATION TO THE ROBOT ACTUATORS
        #self.motors.sendV(10)
        #self.motors.sendW(5)

        #SHOW THE FILTERED IMAGE ON THE GUI
        self.setRightImageFiltered(np.uint8(img_L_filtered))
        self.setLeftImageFiltered(np.uint8(imageLeft))

    def getPoints(self, img, divisions):
        kernel_b = (5,5)
        kernel_c = (3,3)

        # lower_color = np.array([0, 100, 100])
        # upper_color = np.array([180, 255, 255])
        lower_color = np.array([0, 100, 100])
        upper_color = np.array([180, 255, 255])

        blur = cv2.GaussianBlur(img, kernel_b, 0)
        img_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, lower_color, upper_color)
        mask = cv2.erode(mask, kernel= kernel_c, iterations= 2)
        mask = cv2.dilate(mask, kernel= kernel_c, iterations= 2)

        h, w = mask.shape[:2]
        h_m = h // 2
        points = []

        for y in range(h_m, h, h_m // divisions):
            whites = np.nonzero(mask[y, :])
            if len(whites[0]) != 0:
                cv2.line(img, (0, y), (w, y), (255, 0, 0), 1)
                w_tmp = (whites[0][0] + whites[0][-1]) // 2
                cv2.circle(img, (int(w_tmp), int(y)), 2, (0, 255, 0), 3)
                points.append([int(w_tmp), int(y)])

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        return img, mask, points

    def getDistances(self, ref, curr):
        distances = []
        # for