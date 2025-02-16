#Official Autonomous Agent File
import math

import carla
import cv2 as cv
import numpy as np

from geometry.pose import Pose
from leaderboard.autoagents.human_agent import HumanAgent
from localization.robot_tracker import PoseEstimator
from leaderboard.autoagents.autonomous_agent import AutonomousAgent


def get_entry_point():
    return 'AutoAgent'


cameras = [carla.SensorPosition.Front, carla.SensorPosition.FrontLeft, carla.SensorPosition.FrontRight,
            carla.SensorPosition.Left, carla.SensorPosition.Right, carla.SensorPosition.BackLeft,
            carla.SensorPosition.BackRight, carla.SensorPosition.Back]

class AutoAgent(HumanAgent, AutonomousAgent):
    def __init__(self):
        super().__init__()
        self.pose_estimator: PoseEstimator = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.orb = None
        self.bf = None

    def use_fiducials(self):
        return True

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        """ Initialize a counter to keep track of the number of simulation steps. """

        self.frame = -1
        print("Setup AutoAgent")

        # self.stereo = cv.StereoBM.create(numDisparities=320, blockSize=15)
        # self.stereo.setMinDisparity(0)
        # self.stereo.setNumDisparities(320)
        # # self.stereo.setMinDisparity(64)
        # self.stereo.setUniquenessRatio(10)
        # self.stereo.setDisp12MaxDiff(0)
        # self.stereo.setSpeckleWindowSize(50)
        # self.stereo.setSpeckleRange(1)
        # self.stereo.setTextureThreshold(40)

        self.orb = cv.ORB()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.pose_estimator = PoseEstimator(initial_pose=Pose.from_carla(self.get_initial_position()))





    def sensors(self):

        """ In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048)
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light. """
        # Reduced resolution of cameras for better performance


        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 'light_intensity': 0.5, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': True, 'light_intensity': 0.5, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.Left: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.Right: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.BackLeft: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.BackRight: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.Back: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
        }
        return sensors

    def run_step(self, input_data):
        self.frame += 1

        if (self.frame == 0):
            print("Moving drums up")
            self.set_camera_state(carla.SensorPosition.FrontLeft, False)
            self.set_camera_state(carla.SensorPosition.FrontRight, False)
            self.set_front_arm_angle(math.radians(90))
            self.set_back_arm_angle(math.radians(90))

        # if (self.frame == 25):
        #     print("Starting to run")
        #     self.set_camera_state(carla.SensorPosition.FrontLeft, True)
        #     self.set_camera_state(carla.SensorPosition.FrontRight, True)
        # if (self.frame <= 25):
        #     return carla.VehicleVelocityControl()

        # img_l: np.ndarray = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
        # img_r: np.ndarray  = input_data["Grayscale"][carla.SensorPosition.FrontRight]
        #
        # if (img_l is None or img_r is None):
        #     return carla.VehicleVelocityControl()

        # cv.imshow("Left Camera", img_l)
        # cv.imshow("Right Camera", img_r)

        self.pose_estimator.step(timestamp=self.get_mission_time(), imu_data=self.get_imu_data())
        print("estimated pose: " + str(self.pose_estimator.get_fused_pose()) + " actual: " + str(Pose.from_carla(self.get_transform())))
        print("velocity: " + str(self.pose_estimator.get_1d_velocity()) + " / " + str(self.pose_estimator.get_velocity()))

        print("frame: ", self.frame)

        for camera in cameras:
            self.get_camera_position(camera)



        # Calculating disparith using the StereoSGBM algorithm
        # disp = self.stereo.compute(imgL, imgR).astype(np.float32)
        # disp = cv.normalize(disp, 0, 255, cv.NORM_MINMAX)

        # keypoints, descriptors = self.orb.detectAndCompute(img_l, None)
        #
        # if (self.prev_keypoints is not None and self.prev_descriptors is not None):
        #     matches = self.bf.match(self.prev_descriptors, descriptors)
        #     matches = sorted(matches, key=lambda x: x.distance)
        #
        # self.prev_keypoints = keypoints
        # self.prev_descriptors = descriptors
        # cv.waitKey(1)
        return super().run_step(input_data)


    def finalize(self):
        super().finalize()
        cv.destroyAllWindows()
