from typing import Callable, Optional, Dict
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import time
import math

class User:
    def __init__ (self):
        self.pose = {
            "bravo_axis_a": math.pi * 0, # max is around 0.25*math.pi
            "bravo_axis_b": 0,
            "bravo_axis_c": math.pi * 0.25,
            "bravo_axis_d": math.pi * 0,
            "bravo_axis_e": math.pi * 1,
            "bravo_axis_f": math.pi * 1,
            "bravo_axis_g": math.pi * 0.5
        }
        self.inc = -0.07
        self.last_time = time.time()
        self.mode = 0    # 0 is rove mode, 1 is get close mode, 2 is latch mode
        self.is_level = False
        self.start_time = time.time()
        self.follow_dir = 1
        self.reference_midpoint = [300, 200]
        self.reference_distance = 0.035
        self.real_april_dist = 0.272
        self.lastPixelWidth = 50
        self.clawOpenWidth = 15

        return

    def get_dist_and_midpoint(self, camFeed):
        # Define boundaries for the colours we are masking (we want to mask black but should be close to white as we are inverting it)
        lower = [200,200,200]
        upper = [255,255,255]
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # Invert the image and cover up the bottom which contains the timer
        inverted = cv2.bitwise_not(camFeed, camFeed)
        cv2.rectangle(inverted,(0,450),(650,480),(255,0,255),thickness=-1)

        # Create the colour mask to only display black parts of the image
        mask = cv2.inRange(camFeed, lower, upper)
        aprilTags = cv2.bitwise_and(inverted, inverted, mask = mask)
        
        # Convert the image to binary image (via grayscale) and find contours in the binary image
        gray_image = cv2.cvtColor(aprilTags, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray_image,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # Initialise lists of centroids to display
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
        # Tag for if we only see one centroid is also declared here
        calcDistance = -1
        midpoint = -1
        pixelDistance = -1
        loneCentroid = False

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
        elif len(centroids) == 1:
            midpoint = centroids[0]
            loneCentroid = True
        cv2.imshow("Movement Calculations Reference", aprilTags)

        return calcDistance/10, midpoint, pixelDistance, loneCentroid

    def user_defined_inputs(self, globalPoses, send_pose_command):
        newPose = -1
        # Center
        if cv2.waitKey(1) == ord('q'):
            send_pose_command(np.array([50, 0, 0]), np.array([0, 1, 0, 1]))
        
        # Move on x-y
        if cv2.waitKey(1) == ord('w'):
            send_pose_command(np.array([globalPoses['end_effector_joint'][0][0]+0.05, globalPoses['end_effector_joint'][0][1], globalPoses['end_effector_joint'][0][2]]), np.array([0, 1, 0, 1]))
        if cv2.waitKey(1) == ord('a'):
            send_pose_command(np.array([globalPoses['end_effector_joint'][0][0], globalPoses['end_effector_joint'][0][1]+0.05, globalPoses['end_effector_joint'][0][2]]), np.array([0, 1, 0, 1]))
        if cv2.waitKey(1) == ord('s'):
            send_pose_command(np.array([globalPoses['end_effector_joint'][0][0]-0.05, globalPoses['end_effector_joint'][0][1], globalPoses['end_effector_joint'][0][2]]), np.array([0, 1, 0, 1]))
        if cv2.waitKey(1) == ord('d'):
            send_pose_command(np.array([globalPoses['end_effector_joint'][0][0], globalPoses['end_effector_joint'][0][1]-0.05, globalPoses['end_effector_joint'][0][2]]), np.array([0, 1, 0, 1]))
        
        # Move on z
        if cv2.waitKey(1) == ord('x'):
            send_pose_command(np.array([globalPoses['end_effector_joint'][0][0], globalPoses['end_effector_joint'][0][1], globalPoses['end_effector_joint'][0][2]+0.05]), np.array([0, 1, 0, 1]))
        if cv2.waitKey(1) == ord('z'):
            send_pose_command(np.array([globalPoses['end_effector_joint'][0][0], globalPoses['end_effector_joint'][0][1], globalPoses['end_effector_joint'][0][2]-0.05]), np.array([0, 1, 0, 1]))

        return newPose

    def rove_pos_revert(self):
        self.pose = {
            "bravo_axis_a": math.pi * 0,                    # Claw,max is around 0.25*math.pi
            "bravo_axis_b": 0,                              # Claw swivel
            "bravo_axis_c": math.pi * 0.25,                 # Thrird elbow
            "bravo_axis_d": math.pi * 0,                    # Wrist swivel
            "bravo_axis_e": math.pi * 0.9,                  # Second elbow
            "bravo_axis_f": math.pi * 1,                    # First elbow
            "bravo_axis_g": self.pose["bravo_axis_g"]       # Base rotation
        }

    def rove(self, send_pose_command, globalPoses):
        #send_pose_command([400, 0, 0], [0,1,0,1])
        #send_pose_command([500, 0, 0])
        #self.rove_pos_revert()
        if abs(self.inc) != 0.08:
            self.inc = 0.08

        if self.pose["bravo_axis_g"] > 0.75*math.pi:
            self.inc = -0.08
        elif self.pose["bravo_axis_g"] < 0.25*math.pi:
            self.inc = 0.08

        self.pose["bravo_axis_g"] += self.inc

        send_pose_command([math.sin(self.pose["bravo_axis_g"])*500, math.cos(self.pose["bravo_axis_g"]*500), 0])
        
        #send_pose_command([math.sin(math.pi * self.inc + globalPoses[0]), math.cos(math.pi * self.inc + globalPoses[1]), 100])

    def center_centroid(self, send_pose_command, globalPoses, midpoint):
        xPixelOffset = self.reference_midpoint[0] - midpoint[0]
        yPixelOffset = self.reference_midpoint[1] - midpoint[1]

        xDistOffset = (xPixelOffset / self.lastPixelWidth)*self.real_april_dist
        yDistOffset = (yPixelOffset / self.lastPixelWidth)*self.real_april_dist

        newXLoc = globalPoses['end_effector_joint'][0][0]-xDistOffset
        newYLoc = globalPoses['end_effector_joint'][0][1]+yDistOffset
        newZLoc = globalPoses['end_effector_joint'][0][2]+0.1
        
        send_pose_command(np.array([newXLoc, newYLoc, newZLoc]), np.array([0, 1, 0, 1]))
        #newPose["bravo_axis_d"] = math.pi * -0.15
        #self.pose = newPose

    def latch(self, send_pose_command, globalPoses, midpoint, handleDistance, aprilPixelWidth):
        # Pixel width is the length between the two april tags in pixels
        xPixelOffset = self.reference_midpoint[0] - midpoint[0]
        yPixelOffset = self.reference_midpoint[1] - midpoint[1]

        xDistOffset = (xPixelOffset / aprilPixelWidth)*self.real_april_dist
        yDistOffset = (yPixelOffset / aprilPixelWidth)*self.real_april_dist
        zDistOffset = self.reference_distance - handleDistance

        newXLoc = globalPoses['end_effector_joint'][0][0]-xDistOffset
        newYLoc = globalPoses['end_effector_joint'][0][1]+yDistOffset
        newZLoc = globalPoses['end_effector_joint'][0][2]+zDistOffset

        # print(xDistOffset,yDistOffset,zDistOffset)

        send_pose_command(np.array([newXLoc, newYLoc, newZLoc]), np.array([0, 1, 0, 1]))
        #self.pose = newPose

        self.lastPixelWidth = aprilPixelWidth

    def run(self,
            image: np.ndarray, 
            global_poses: Dict[str, np.ndarray],
            send_pose_command: Callable[[np.ndarray, Optional[np.ndarray]], None],
            send_jaw_cmd: Callable[[float], None]
        ):
    
        #send_pose_command(np.array([50, 0, 0]), np.array([0, 1, 0, 1]))
        #send_pose_command([400, -10, 60], [0, 1, 0, 1])
        #print(global_poses)
        send_jaw_cmd(15)
        #print("hey")
        #cv2.resize(image, (480, 640), interpolation= cv2.INTER_LINEAR)
        cv2.imshow("View", image)   # Image is 480 (height), by 640 (width)
        cv2.waitKey(1)

        
        camToHandleDist, handlePoint, pixelWidth, loneCentroid = self.get_dist_and_midpoint(image)

        # Check conditions to determine right mode

        self.mode = 0
        self.mode = 0
        if handlePoint != -1:
            if loneCentroid:
                self.mode = 1
            else:
                self.mode = 2

        # If in rove mode
        if self.mode == 0:
            self.rove(send_pose_command, global_poses)
        elif self.mode == 1:
            self.center_centroid(send_pose_command, global_poses,  handlePoint)
        elif self.mode == 2:
            self.latch(send_pose_command, global_poses,  handlePoint, camToHandleDist, pixelWidth)
            if camToHandleDist < 0.045:
                send_jaw_cmd(0)
            else:
                send_jaw_cmd(self.clawOpenWidth)

        # Getting inputs for manual overide
        userDefPose = self.user_defined_inputs(global_poses, send_pose_command)
        if userDefPose != -1:
            self.pose = userDefPose

        return self.pose

        """Run loop to control the Bravo manipulator.

        Parameters
        ----------
        image: np.ndarray
            The latest camera image frame. Colour order is RGB.

        global_poses: Dict[str, np.ndarray]
            A dictionary with the global camera and end-effector pose. The keys are
            'camera_end_joint' and 'end_effector_joint'. Each pose consists of a 3x1
            position (x, y, z) in mm and a 4x1 quaternion (x, y, z, w) defining the 
            orientation.
        
        send_pose_command: function, (pos: np.ndarray, orient: np.ndarray = None)
            Function to send command to the Bravo to move the end-effector to the given
            position (x, y, z) in mm and quaternion orientation (x, y, z, w) in the global
            reference frame. If orientation argument is not provided, the end-effector
            will try to do pure translation.

        send_jaw_cmd: function, (pos: float)
            Function to send command to the Bravo to close or open the jaws by the distance
            given as a single float in mm, where 0 is fully closed.
        """

        # Flood your terminal with end-effector and camera pose data
        # print(f"END-EFFECTOR:\n     position: {global_poses['end_effector_joint'][0]}" 
        #                  + f"\n  orientation: {global_poses['end_effector_joint'][1]}")
        # print(f"WRIST CAMERA:\n     position: {global_poses['camera_end_joint'][0]}" 
        #                  + f"\n  orientation: {global_poses['camera_end_joint'][1]}")

        # Send a global pose command to the Bravo
        # send_pose_command([450, -320,  127], [0, 0, 0, 1])
        
        # Send jaw command to Bravo
        # send_jaw_cmd(30)

        # Preview camera stream
        # bgrFrame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # needs BGR to preview correctly
        # cv2.imshow("Camera Stream", bgrFrame)
        # cv2.waitKey(1)

        pass