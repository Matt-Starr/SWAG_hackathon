""" User code lives here """
import time
from typing import Dict
import math
from typing import Callable, Optional
import numpy as np
import cv2
from numpy.lib.type_check import imag, mintypecode

class User:
    def __init__(self) -> None:
        self.pose = {
            "bravo_axis_a": math.pi * 0, # max is around 0.25*math.pi
            "bravo_axis_b": 0,
            "bravo_axis_c": math.pi * 0.25,
            "bravo_axis_d": math.pi * 0,
            "bravo_axis_e": math.pi * 1,
            "bravo_axis_f": math.pi * 1,
            "bravo_axis_g": math.pi
        }
        self.inc = 0.08
        self.last_time = time.time()
        self.mode = 0    #0 is rove mode, 1 is get close mode, 2 is latch mode
        self.is_level = False
        self.time = time.time()

        return

    def get_dist_and_midpoint(self, camFeed):
        # define boundaries for the colours we are masking (we want to mask black but should be close to white as we are inverting it)
        lower = [230,230,230]
        upper = [255,255,255]
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # invert the image and cover up the bottom which contains the timer
        inverted = cv2.bitwise_not(camFeed, camFeed)
        cv2.rectangle(inverted,(0,450),(650,480),(255,0,255),thickness=-1)

        # Create the colour mask to only display black parts of the image
        mask = cv2.inRange(camFeed, lower, upper)
        aprilTags = cv2.bitwise_and(inverted, inverted, mask = mask)
        
        # convert the image to binary image (via grayscale) and find contours in the binary image
        gray_image = cv2.cvtColor(aprilTags, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray_image,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # initialise lists of centroids to display
        centroids = []

        for c in contours:
            # Calculate moments for each contour
            M = cv2.moments(c)

            # Calculate x,y coordinate of each centroid
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Check if there is already a centroid placed close to the one we intend to place, if there is it is likely that its the same shape so don't include
                check = False
                if len(centroids) > 0:
                    for existingCentroid in centroids:
                        if abs(existingCentroid[0] - cX) < 50 and abs(existingCentroid[1] - cY) < 50:
                            check = True

                # If we dont find a similarly place centroid add the new centroid to the list 
                if check == False:
                    centroids.append([cX,cY])

        # Display centroids
        for i in centroids:
            cv2.circle(aprilTags, (i[0], i[1]), 5, (255, 0, 255), -1)
            positionString = str(i[0]) + "," + str(i[1])
            cv2.putText(aprilTags, positionString, (i[0] - 25, i[1] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Initialise values for these incase we can't determine a line, a negative output should be interpreted as a failure
        calcDistance = -1
        midpoint = -1

        # Calculate distance between centroids in terms of pixels and find the coordinates of the midpoint
        if len(centroids) == 2:
            image = cv2.line(aprilTags, centroids[0], centroids[1], (255,0,0), 2)
            pixelDistance = (abs(centroids[0][0] - centroids[1][0])**2 + abs(centroids[0][1] - centroids[1][1])**2)**(1/2)
            cv2.putText(aprilTags, f"Pixel Width = {pixelDistance:.2f} pixels", (200,450),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            midpoint = [(centroids[0][0] + centroids[1][0])/2, (centroids[0][1] + centroids[1][1])/2]
            cv2.circle(aprilTags, (int(midpoint[0]), int(midpoint[1])), 5, (0, 255, 0), -1)
            midpointString = str(int(midpoint[0])) + "," + str(int(midpoint[1]))
            cv2.putText(aprilTags, midpointString, (int(midpoint[0]) - 25, int(midpoint[1]) + 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            refPixels = 221.94
            measuredDist = 0.375
            realWidth = 0.272
            focalLen = (refPixels*measuredDist)/realWidth
            calcDistance = (realWidth*focalLen)/pixelDistance
            cv2.putText(aprilTags, f"Calculation = {calcDistance*100:.2f} mm", (200,420),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("apriltags", aprilTags)

        return calcDistance/10, midpoint

    def user_defined_inputs(self, globalPoses, calcIK):
        newPose = -1
        #center
        if cv2.waitKey(1) == ord('q'):
            newPose = calcIK(np.array([0.5, 0, 0]), np.array([0, 1, 0, 1]))
        
        #move on x-y
        if cv2.waitKey(1) == ord('w'):
            newPose = calcIK(np.array([globalPoses['end_effector_joint'][0][0]+0.05, globalPoses['end_effector_joint'][0][1], globalPoses['end_effector_joint'][0][2]]), np.array([0, 1, 0, 1]))
        if cv2.waitKey(1) == ord('a'):
            newPose = calcIK(np.array([globalPoses['end_effector_joint'][0][0], globalPoses['end_effector_joint'][0][1]+0.05, globalPoses['end_effector_joint'][0][2]]), np.array([0, 1, 0, 1]))
        if cv2.waitKey(1) == ord('s'):
            newPose = calcIK(np.array([globalPoses['end_effector_joint'][0][0]-0.05, globalPoses['end_effector_joint'][0][1], globalPoses['end_effector_joint'][0][2]]), np.array([0, 1, 0, 1]))
        if cv2.waitKey(1) == ord('d'):
            newPose = calcIK(np.array([globalPoses['end_effector_joint'][0][0], globalPoses['end_effector_joint'][0][1]-0.05, globalPoses['end_effector_joint'][0][2]]), np.array([0, 1, 0, 1]))
        
        #move on z
        if cv2.waitKey(1) == ord('x'):
            newPose = calcIK(np.array([globalPoses['end_effector_joint'][0][0], globalPoses['end_effector_joint'][0][1], globalPoses['end_effector_joint'][0][2]+0.05]), np.array([0, 1, 0, 1]))
        if cv2.waitKey(1) == ord('z'):
            newPose = calcIK(np.array([globalPoses['end_effector_joint'][0][0], globalPoses['end_effector_joint'][0][1], globalPoses['end_effector_joint'][0][2]-0.05]), np.array([0, 1, 0, 1]))

        return newPose

    def rove(self):
        # self.pose["bravo_axis_g"] += self.inc
        # if (self.pose["bravo_axis_g"] >= 2 * math.pi):
        #     self.pose["bravo_axis_g"] -= 4 * math.pi
        if self.pose["bravo_axis_d"] > 0.5*math.pi:
            self.inc = -0.08
        elif self.pose["bravo_axis_d"] < -0.5*math.pi:
            self.inc = 0.08
        
        self.pose["bravo_axis_d"] += self.inc

    def get_close(self):
        pass

    def latch(self):
        pass

    def run(self,
            image: list, 
            global_poses: Dict[str, np.ndarray],
            calcIK: Callable[[np.ndarray, Optional[np.ndarray]], Dict[str, float]],
            ) -> Dict[str, float]:
        
        cv2.imshow("View", image)
        #Image is 480 (height), by 640 (width)
        cv2.waitKey(1)

        #print(image.shape())

        camToHandleDist, handlePoint = self.get_dist_and_midpoint(image)

        #Check conditions to determine right mode
        self.mode = 0
        if handlePoint != -1:
            print(camToHandleDist)
            if camToHandleDist > 0.07:
                self.mode = 1
            else:
                self.mode = 2

        # If in rove mode
        if self.mode == 0:
            print('0')
            self.rove()
        elif self.mode == 1:
            print('1')
            self.get_close()
        elif self.mode == 2:
            print('2')
            self.latch()


        # Getting inputs for manual control
        userDefPose = self.user_defined_inputs(global_poses, calcIK)
        if userDefPose != -1:
            self.pose = userDefPose
        #print(f"{time.time() - self.time:.2f}")
        
        return self.pose