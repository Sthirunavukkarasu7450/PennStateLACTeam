import numpy as np
from carla.libcarla import Transform
from pyquaternion import Quaternion

from geometry.pose import Pose, PTranslation

history_time = 1

gravity = PTranslation(0, 0, -1.6220)


class TimestampedPose:
    def __init__(self, timestamp: float, pose: Pose):
        self.timestamp = timestamp
        self.pose = pose


class PoseEstimator:
    def __init__(self, initial_pose: Pose = Pose(PTranslation(0, 0, 0), Quaternion()),
                 initial_velocity: PTranslation = PTranslation(0, 0, 0),
                 timestamp: float = 0.0,
                 ):
        self.current_pose = initial_pose
        self.current_velocity = initial_velocity
        self.last_timestamp = timestamp

        self.pose_history: [TimestampedPose] = []

        self.pose_history.append(TimestampedPose(timestamp, initial_pose))

    def get_fused_pose(self) -> Pose:
        return self.current_pose

    def get_velocity(self) -> PTranslation:
        return self.current_velocity

    def get_1d_velocity(self) -> float:
        return self.current_velocity.magnitude()

    def step(self, timestamp: float, imu_data: np.ndarray):
        """
        timestamp: seconds since epoch
        imu_data: [x, y, z, roll, pitch, yaw] (m/s^2, rad/s)
        """

        if not isinstance(imu_data, np.ndarray) or imu_data.shape[0] != 6:
            raise ValueError("imu_data must be a numpy array of shape (6,)")

        dt = timestamp - self.last_timestamp
        if dt <= 0:
            return  # Skip this step if time has not progressed

        # Create the rotational rate quaternion
        rot_rate = np.array([imu_data[3], imu_data[4], imu_data[5]])
        self.current_pose.orientation.integrate(rot_rate, dt)

        # Rotate acceleration to global frame
        accel = PTranslation(imu_data[0], imu_data[1], imu_data[2])
        accel = accel.rotate_by(self.current_pose.orientation)
        print("accel " + str(accel))
        accel =  accel + gravity
        print("accel + gravity " + str(accel))


        # Update velocity and position
        self.current_velocity += accel * dt
        self.current_pose.position += self.current_velocity * dt

        # Update last timestamp
        self.last_timestamp = timestamp

        pose_history_entry = TimestampedPose(timestamp, self.current_pose.copy())
        self.pose_history.append(pose_history_entry)
        if (self.pose_history[-1].timestamp - self.pose_history[0].timestamp) > history_time:
            self.pose_history.pop(0)