import numpy as np

from controller import Supervisor

from alamaklib import (
    IMU,
    AckermannSteering,
    AlamakSupervisor,
    CameraWrapper,
    Encoder,
    Motors,
    decompose_acceleration,
    rotate_to_world_frame,
    estimate_steering_angle_deg
)
from data import AlamakData, AlamakDataGT, AlamakParams
from plots import save_plots

# Setup
# -----------------------------------------------------------------------------

robot = Supervisor()
sup = AlamakSupervisor(robot)

# TIME_STEP = int(robot.getBasicTimeStep())
TIME_STEP = 16
TIME_STEP_SEC = TIME_STEP / 1000.0

ackermann = AckermannSteering(robot.getDevice("fl_servo"), robot.getDevice("fr_servo"))
motors = Motors(left=robot.getDevice("rl_motor"), right=robot.getDevice("rr_motor"))
rl_encoder = Encoder(robot.getDevice("rl_encoder"), TIME_STEP)
rr_encoder = Encoder(robot.getDevice("rr_encoder"), TIME_STEP)
fl_servo_encoder = Encoder(robot.getDevice("fl_servo_encoder"), TIME_STEP)
fr_servo_encoder = Encoder(robot.getDevice("fr_servo_encoder"), TIME_STEP)
camera = CameraWrapper(robot.getDevice("rpi_camera_v2"), TIME_STEP)
imu = IMU(
    gyro=robot.getDevice("imu_gyro"),
    inertial_unit=robot.getDevice("imu_inertial_unit"),
    accel=robot.getDevice("imu_accel"),
    compass=robot.getDevice("imu_compass"),
    sampling_period_ms=TIME_STEP,
)

# Scenarios
# 1. Straight line
# 2. Circle
# 3. Figure 8

# Data
# -----------------------------------------------------------------------------

STEPS_TO_PLOT = 1000
TARGET_VELOCITY = 20  # [rad/s]
TARGET_STEERING_ANGLE = 10  # deg

data = AlamakData()
data_gt = AlamakDataGT()

# Controller
# -------------------------------------------------------------------------------------------------

# Skip initial simulation frames (filters out initial physics collisions and noise etc.)
for i in range(3):
    robot.step(TIME_STEP)

step = 0
motors.velocity = TARGET_VELOCITY
ackermann.angle = TARGET_STEERING_ANGLE


class Speedometer:
    """Calculates vehicle speed from encoders"""

    def __init__(self, wheel_radius: float, init_angle: float):
        self._wheel_radius = wheel_radius
        self._last_angle = init_angle

    @staticmethod
    def overflow_corrected_diff(current: float, previous: float) -> float:
        return (current - previous + np.pi) % (2 * np.pi) - np.pi

    def get_speed(self, angle: float, dt: float) -> float:
        angle_diff = Speedometer.overflow_corrected_diff(angle, self._last_angle)
        angular_speed = angle_diff / dt
        self._last_angle = angle
        return angular_speed * self._wheel_radius


rl_speedometer = Speedometer(AlamakParams.WHEEL_RADIUS, rl_encoder.value_rad)
rr_speedometer = Speedometer(AlamakParams.WHEEL_RADIUS, rr_encoder.value_rad)

class LowPassFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = 0.0
        
    def update(self, measurement):
        self.value = self.alpha * measurement + (1.0 - self.alpha) * self.value
        return self.value

yaw_rate_filter = LowPassFilter(alpha=0.55)

while robot.step(TIME_STEP) != -1:
    # Smooth steering angle as a function of step
    # ackermann.angle = TARGET_STEERING_ANGLE * np.sin(step * 0.01)

    data.target_steering_angle.append(ackermann.angle)
    data.target_speed.append(motors.velocity)

    # Ground Truth
    v_xyz = sup.velocity[:3]
    data_gt.v_xyz.append(v_xyz)
    data_gt.speed.append(sup.speed)
    
    a_xyz = (data_gt.v_xyz[step] - data_gt.v_xyz[step - 1]) / TIME_STEP_SEC
    data_gt.a_xyz.append(a_xyz)
    data_gt.a.append(np.linalg.norm(a_xyz))

    a_tangential, a_normal = decompose_acceleration(v_xyz, a_xyz)
    data_gt.a_tangential.append(a_tangential)
    data_gt.a_normal.append(a_normal)
    
    data_gt.position_xyz.append(sup.position)
    data_gt.rotation_xyz.append(sup.rotation_euler_angles)
    fl_angle = fl_servo_encoder.value_deg
    fr_angle = fr_servo_encoder.value_deg
    
    # Convert from 0-360 to -180 to 180
    if fl_angle > 180:
        fl_angle -= 360
    if fr_angle > 180:
        fr_angle -= 360
    
    data_gt.wheel_steering_angles.append((fl_angle, fr_angle))

    # Estimated
    rotation_xyz_imu = imu.get_euler_angles()
    data.rotation_xyz_imu.append(rotation_xyz_imu)
    
    a_xyz_imu_local_frame = imu.get_linear_acceleration()
    a_xyz_imu = rotate_to_world_frame(a_xyz_imu_local_frame, rotation_xyz_imu)
    data.a_xyz_imu.append(a_xyz_imu)
    data.a_imu.append(np.linalg.norm(a_xyz_imu))

    v_xyz_imu = np.zeros(3) if step == 0 else data.v_xyz_imu[step - 1] + a_xyz_imu * TIME_STEP_SEC
    data.v_xyz_imu.append(v_xyz_imu)
    a_tang_imu, a_norm_imu = decompose_acceleration(v_xyz_imu, a_xyz_imu)
    data.a_tangential_imu.append(a_tang_imu)
    data.a_normal_imu.append(a_norm_imu)
    
    speed_imu = 0.0 if step == 0 else data.speed_imu[step - 1] + a_tang_imu * TIME_STEP_SEC
    data.speed_imu.append(speed_imu)
    speed_rl_encoder = rl_speedometer.get_speed(angle=rl_encoder.value_rad, dt=TIME_STEP_SEC)
    speed_rr_encoder = rr_speedometer.get_speed(angle=rr_encoder.value_rad, dt=TIME_STEP_SEC)
    speed_encoders = (speed_rl_encoder + speed_rr_encoder) / 2
    data.speed_encoders.append(speed_encoders)
    data.speed_wheels.append((speed_rl_encoder, speed_rr_encoder))
    
    angular_v_xyz = imu.get_angular_velocity()
    data.angular_v_xyz_imu.append(angular_v_xyz)
    steering_angle_estimate = estimate_steering_angle_deg(speed=speed_encoders, yaw_rate=angular_v_xyz[2])
    data.steering_angle_estimate.append(steering_angle_estimate)

    if step == STEPS_TO_PLOT:
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        alamak_data_path = "data/alamak_data.pkl"
        alamak_data_gt_path = "data/alamak_data_gt.pkl"
        data.save(alamak_data_path)
        data_gt.save(alamak_data_gt_path)
        data_loaded: AlamakData = AlamakData.load(alamak_data_path)
        data_gt_loaded: AlamakDataGT = AlamakDataGT.load(alamak_data_gt_path)
        save_plots(data_loaded, data_gt_loaded)
        break
        # robot.simulationReset()
        # robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
    step += 1
