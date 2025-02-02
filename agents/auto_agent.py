#Official Autonomous Agent File

from leaderboard.autoagents.autonomous_agent.AutonomousAgent import AutonomousAgent # type: ignore
import gtsam
from gtsam.symbol_shorthand import X
import numpy as np
def get_entry_point():
    return 'AutoAgent'


class AutoAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        # Initialize 6DoF pose using documentation-specified methods
        init_transform = self.get_initial_position()
        self.current_pose = gtsam.Pose3(
            gtsam.Rot3.Ypr(init_transform.rotation.yaw,
                          init_transform.rotation.pitch,
                          init_transform.rotation.roll),
            np.array([init_transform.location.x,
                     init_transform.location.y,
                     init_transform.location.z])
        )
        
        # 6DoF factor graph setup
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]))  # x,y,z,roll,pitch,yaw

        # IMU calibration from documentation
        self.gyro_bias = np.array([0.0021, 0.0017, 0.003])
        self.accel_bias = np.array([0.014, 0.022, 0.018])

    def sensors(self):
        return {
            carla.SensorPosition.Front: {
                'camera_active': True, 'width': 2048, 'height': 1536,
                'use_semantic': False},
            carla.SensorPosition.FrontLeft: {
                'camera_active': True, 'width': 2048, 'height': 1536,
                'use_semantic': False},
            carla.SensorPosition.Left: {
                'camera_active': True, 'width': 1024, 'height': 768,
                'use_semantic': False}
        }

    def update_odometry(self, imu_data, dt):
        # Full 6DoF IMU integration
        raw_accel = np.array(imu_data[:3]) - self.accel_bias
        raw_gyro = np.array(imu_data[3:]) - self.gyro_bias
        
        # Convert to global frame using current orientation
        R = self.current_pose.rotation().matrix()
        global_accel = R @ raw_accel - np.array([0, 0, 1.625])  # Lunar gravity
        
        # Integrate motion
        delta_pose = gtsam.Pose3(
            gtsam.Rot3.Expmap(raw_gyro * dt),
            global_accel * dt**2 / 2
        )
        self.current_pose = self.current_pose.compose(delta_pose)

    def camera_to_global(self, u, v, depth):
        # 6DoF projection using documentation-specified camera geometry
        fx = fy = 1024  # From camera specs
        cx, cy = 1024, 768
        
        # Camera to body transform (from geometry section)
        cam_position = gtsam.Pose3(
            gtsam.Rot3(), 
            np.array([0.28, 0.081, 0.131])  # FrontLeft camera offset
        )
        
        # Pixel to global coordinates
        z = depth[v,u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        point_body = cam_position.transformFrom(gtsam.Point3(x, y, z))
        return self.current_pose.transformFrom(point_body)

    def update_elevation_map(self, disparity):
        depth = (1024 * 0.162) / (disparity + 1e-6)  # From stereo baseline
        height_map = self.geometric_map.get_map_array()
        
        for x_idx in range(disparity.shape[1]):
            for y_idx in range(disparity.shape[0]):
                if disparity[y_idx, x_idx] > 0:
                    global_pos = self.camera_to_global(x_idx, y_idx, depth)
                    self.geometric_map.set_height(
                        global_pos[0], global_pos[1], global_pos[2])

    def run_step(self, input_data):
        # Full 6DoF processing pipeline
        imu = self.get_imu_data()
        current_time = self.get_mission_time()
        
        if self.last_time is not None:
            dt = current_time - self.last_time
            self.update_odometry(imu, dt)
            self.graph.add(gtsam.BetweenFactorPose3(
                X(self.step_count-1), X(self.step_count), 
                self.current_pose, self.odometry_noise))
            
        # Update map and plan motion
        if input_data['Grayscale'][carla.SensorPosition.Front] is not None:
            disparity = self.calculate_disparity(...)
            self.update_elevation_map(disparity)
            
        # Return control commands
        print(f"Step {self.step_count}: "f"Pos={self.current_pose[:2].round(2)}, "f"Cells mapped={self.geometric_map.mapped_cells}")
        return self.spiral_coverage_pattern()

    def align_with_charger(self):
        # 6DoF alignment using documentation-specified coordinates
        target_pose = gtsam.Pose3(
            gtsam.Rot3.Ypr(0, 0, 0),
            np.array([0, 1.452, 0.509])  # Charger position from docs
        )
        
        while True:
            error = self.current_pose.localCoordinates(target_pose)
            control = self.pid_controller(error)
            yield carla.VehicleVelocityControl(control[0], control[1])
    def update_elevation_map(self, disparity_map):
        focal_length = 1024  # px (from camera specs)
        baseline = 0.162  # meters (front stereo pair)
        depth = (focal_length * baseline) / disparity
        
        # Convert depth to elevation using camera geometry
        for x_idx, y_idx in visible_cells:
            elevation = calculate_cell_elevation(depth, x_idx, y_idx)
            self.geometric_map.set_cell_height(x_idx, y_idx, elevation)
    def detect_rocks(self, image):
        # Edge detection + contour analysis
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL)
        
        for contour in contours:
            if cv2.contourArea(contour) > MIN_ROCK_AREA:
                x, y = convert_to_map_coordinates(contour)
                self.geometric_map.set_cell_rock(x, y, True)
