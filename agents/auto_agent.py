#Official Autonomous Agent File
import math

import carla
import cv2 as cv
import numpy as np
from pynput import keyboard

from leaderboard.autoagents.autonomous_agent import AutonomousAgent

def get_entry_point():
    return 'AutoAgent'


class AutoAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        #dont need listeners since code should be autonomous
        #listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        #listener.start()

        #time step inint

        self.time_step = 0

        """ Add some attributes to store values for the target linear and angular velocity. """

        self.current_v = 0
        self.current_w = 0

        """ Initialize a counter to keep track of the number of simulation steps. """

        self.frame = -1
        print("Setup AutoAgent")

        # Creating an object of StereoSGBM algorithm
        self.stereo = cv.StereoBM.create(numDisparities=256, blockSize=15)
        self.stereo.setMinDisparity(0)
        self.stereo.setNumDisparities(256)
        self.stereo.setMinDisparity(0)

    #no fudicials to maximize points 
    def use_fiducials(self):
        return False


    def sensors(self):

        """ In the sensors method, we define the desired resolution of our cameras (remember that the maximum resolution available is 2448 x 2048)
        and also the initial activation state of each camera and light. Here we are activating the front left camera and light. """
        # Reduced resolution of cameras for better performance


        sensors = {
            carla.SensorPosition.Front: {
                'camera_active': False, 'light_intensity': 0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720'
            },
            carla.SensorPosition.FrontRight: {
                'camera_active': True, 'light_intensity': 1.0, 'width': '1280', 'height': '720'
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

        if (self.frame == 0 and self.time_step == 0):
            print("Moving drums up")
            self.set_camera_state(carla.SensorPosition.FrontLeft, False)
            self.set_camera_state(carla.SensorPosition.FrontRight, False)
            self.set_front_arm_angle(math.radians(90))
            self.set_back_arm_angle(math.radians(90))

        if (self.frame == 25):
            print("Starting to run")
            self.set_camera_state(carla.SensorPosition.FrontLeft, True)
            self.set_camera_state(carla.SensorPosition.FrontRight, True)
        if (self.frame <= 25):
            return carla.VehicleVelocityControl(0,0)




        imgL = input_data["Grayscale"][carla.SensorPosition.FrontLeft]
        imgR = input_data["Grayscale"][carla.SensorPosition.FrontRight]

        if (imgL is None or imgR is None):
            return carla.VehicleVelocityControl()

        cv.imshow("Left Camera", imgL)
        cv.imshow("Right Camera", imgR)
        print("frame: ", self.frame, "current_v: ", self.current_v, "current_w: ", self.current_w, "")





        # Calculating disparith using the StereoSGBM algorithm
        disp = self.stereo.compute(imgL, imgR).astype(np.float32)
        disp = cv.normalize(disp, 0, 255, cv.NORM_MINMAX)
        cv.imshow("Disparity Map", disp)
        cv.waitKey(1)
        return carla.VehicleVelocityControl()


    def finalize(self):
        cv.destroyAllWindows()

        """
        Cleanup
        """
        if hasattr(self, '_hic') and not self._has_quit:
            self._hic.set_black_screen()
            self._hic.quit()
            self._has_quit = True

    #code belwo is for the listner which arent being used since rover autonomous 
    '''
    def on_press(self, key):

        """ This is the callback executed when a key is pressed. If the key pressed is either the up or down arrow, this method will add
        or subtract target linear velocity. If the key pressed is either the left or right arrow, this method will set a target angular
        velocity of 0.6 radians per second. """

        if key == keyboard.Key.up:
            self.current_v += 0.1
            self.current_v = np.clip(self.current_v, 0, 0.3)
        if key == keyboard.Key.down:
            self.current_v -= 0.1
            self.current_v = np.clip(self.current_v, -0.3, 0)
        if key == keyboard.Key.left:
            self.current_w = 0.6
        if key == keyboard.Key.right:
            self.current_w = -0.6

    def on_release(self, key):

        """ This method sets the angular or linear velocity to zero when the arrow key is released. Stopping the robot. """

        if key == keyboard.Key.up:
            self.current_v = 0
        if key == keyboard.Key.down:
            self.current_v = 0
        if key == keyboard.Key.left:
            self.current_w = 0
        if key == keyboard.Key.right:
            self.current_w = 0

        """ Press escape to end the mission. """
        if key == keyboard.Key.esc:
            self.mission_complete()
            cv.destroyAllWindows()
    '''