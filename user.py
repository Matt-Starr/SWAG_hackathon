""" User code lives here """
import time
from typing import Dict
import math
from typing import Callable, Optional
import numpy as np
import cv2
from numpy.lib.type_check import imag

def get_dist_and_midpoint(image):
    # define boundaries for the colours we are masking (we want to mask black but should be close to white as we are inverting it)
    lower = [230,230,230]
    upper = [255,255,255]
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    # invert the image and cover up the bottom which contains the timer
    inverted = cv2.bitwise_not(image, image)
    cv2.rectangle(inverted,(0,450),(650,480),(255,0,255),thickness=-1)

    # Create the colour mask to only display black parts of the image
    mask = cv2.inRange(image, lower, upper)
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

            # If we dont find a similarly place centroid add the new centroid to the list and display
            if check == False:
                cv2.circle(aprilTags, (cX, cY), 5, (255, 0, 255), -1)
                positionString = str(cX) + "," + str(cY)
                cv2.putText(aprilTags, positionString, (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                centroids.append([cX,cY])
    
    # Initialise values for these incase we can't determine a line, a negative output should be interpreted as a failure
    pixelDistance = -1
    midpoint = -1

    # Calculate distance between centroids in terms of pixels and find the coordinates of the midpoint
    if len(centroids) == 2:
        image = cv2.line(aprilTags, centroids[0], centroids[1], (255,0,0), 2)
        pixelDistance = (abs(centroids[0][0] - centroids[1][0])**2 + abs(centroids[0][1] - centroids[1][1])**2)**(1/2)
        cv2.putText(aprilTags, f"Distance = {pixelDistance:.2f} pixels", (200,430),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        midpoint = [(centroids[0][0] + centroids[1][0])/2, (centroids[0][1] + centroids[1][1])/2]
        cv2.circle(aprilTags, (int(midpoint[0]), int(midpoint[1])), 5, (0, 255, 0), -1)
    cv2.imshow("apriltags", aprilTags)

    return pixelDistance, midpoint


class User:
    def __init__(self) -> None:
        self.pose = {
            "bravo_axis_a": math.pi * 0, # max is around 0.25*math.pi
            "bravo_axis_b": 0,
            "bravo_axis_c": math.pi * 0.25,
            "bravo_axis_d": math.pi * 1,
            "bravo_axis_e": math.pi * 1,
            "bravo_axis_f": math.pi * 1,
            "bravo_axis_g": math.pi
        }
        self.inc = 0.08
        self.last_time = time.time()
        self.rove = True

        return


    def run(self,
            image: list, 
            global_poses: Dict[str, np.ndarray],
            calcIK: Callable[[np.ndarray, Optional[np.ndarray]], Dict[str, float]],
            ) -> Dict[str, float]:
        """Run loop to control the Bravo manipulator.

        Parameters
        ----------
        image: list
            The latest camera image frame.

        global_poses: Dict[str, np.ndarray]
            A dictionary with the global camera and end-effector pose. The keys are
            'camera_end_joint' and 'end_effector_joint'. Each pose consitst of a (3x1)
            position (vec3) and (4x1) quaternion defining the orientation.
        
        calcIK: function, (pos: np.ndarray, orient: np.ndarray = None) -> Dict[str, float]
            Function to calculate inverse kinematics. Provide a desired end-effector
            position (vec3) and an orientation (quaternion) and it will return a pose
            dictionary of joint angles to approximate the pose.
        """
        
        cv2.imshow("View", image)
        #Image is 480 (height), by 640 (width)
        cv2.waitKey(1)

        camToHandleDist, handlePoint = get_dist_and_midpoint(image)

        if self.rove:
            # self.pose["bravo_axis_g"] += self.inc
            # if (self.pose["bravo_axis_g"] >= 2 * math.pi):
            #     self.pose["bravo_axis_g"] -= 4 * math.pi
            if self.pose["bravo_axis_d"] > 0.5*math.pi:
                self.inc = -0.08
            elif self.pose["bravo_axis_d"] < -0.5*math.pi:
                self.inc = 0.08
            
            self.pose["bravo_axis_d"] += self.inc

        #center
        if cv2.waitKey(1) == ord('q'):
            self.pose = calcIK(np.array([0.5, 0, 0]), np.array([0, 1, 0, 1]))
            
        #move on x-y
        if cv2.waitKey(1) == ord('w'):
            self.pose = calcIK(np.array([global_poses['end_effector_joint'][0][0]+0.05, global_poses['end_effector_joint'][0][1], global_poses['end_effector_joint'][0][2]]), np.array([0, 1, 0, 1]))
        if cv2.waitKey(1) == ord('a'):
            self.pose = calcIK(np.array([global_poses['end_effector_joint'][0][0], global_poses['end_effector_joint'][0][1]+0.05, global_poses['end_effector_joint'][0][2]]), np.array([0, 1, 0, 1]))
        if cv2.waitKey(1) == ord('s'):
            self.pose = calcIK(np.array([global_poses['end_effector_joint'][0][0]-0.05, global_poses['end_effector_joint'][0][1], global_poses['end_effector_joint'][0][2]]), np.array([0, 1, 0, 1]))
        if cv2.waitKey(1) == ord('d'):
            self.pose = calcIK(np.array([global_poses['end_effector_joint'][0][0], global_poses['end_effector_joint'][0][1]-0.05, global_poses['end_effector_joint'][0][2]]), np.array([0, 1, 0, 1]))
        
        #move on z
        if cv2.waitKey(1) == ord('x'):
            self.pose = calcIK(np.array([global_poses['end_effector_joint'][0][0], global_poses['end_effector_joint'][0][1], global_poses['end_effector_joint'][0][2]+0.05]), np.array([0, 1, 0, 1]))
        if cv2.waitKey(1) == ord('z'):
            self.pose = calcIK(np.array([global_poses['end_effector_joint'][0][0], global_poses['end_effector_joint'][0][1], global_poses['end_effector_joint'][0][2]-0.05]), np.array([0, 1, 0, 1]))

        return self.pose