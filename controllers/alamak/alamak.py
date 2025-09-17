import numpy as np
from time import time_ns

from controller import Supervisor

from alamaklib import (
    IMU,
    AckermannSteering,
    AlamakSupervisor,
    CameraWrapper,
    Encoder,
    Motors,
    Speedometer,
    decompose_acceleration,
    rotate_to_world_frame,
    estimate_steering_angle_deg,
)
from data import AlamakData, AlamakDataGT, AlamakParams
from plots import save_plots

from datalink.ipc import SPMCQueue, MPMCQueue, AddrType
from datalink.data import (
    RealData,
    SensorFusionData,
    ActuatorsData,
    ControlData,
    JPGImageData,
    SpeedometerData,
    IMU2Data,
    EncoderData
)

# Datalink
# -----------------------------------------------------------------------------

# q_state_handle = SPMCQueue("q_state", type=AddrType.TCP, port=11001)
q_state_handle = MPMCQueue("q_state", addr_type=AddrType.TCP, ports=(11001, 11002))
q_control_handle = SPMCQueue("q_control", AddrType.TCP, port=10050)
q_state = q_state_handle.get_producer()
q_control = q_control_handle.get_consumer()

# Setup
# -----------------------------------------------------------------------------

robot = Supervisor()
sup = AlamakSupervisor(robot)

# TIME_STEP = int(robot.getBasicTimeStep())
TIME_STEP: int = 16
TIME_STEP_SEC = TIME_STEP / 1000.0
TIME_STEP_NSEC: int = int(TIME_STEP * 1e6)

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

rl_speedometer = Speedometer(AlamakParams.WHEEL_RADIUS, rl_encoder.value_rad)
rr_speedometer = Speedometer(AlamakParams.WHEEL_RADIUS, rr_encoder.value_rad)

# Data
# -----------------------------------------------------------------------------

STEPS_TO_PLOT = 350 / (TIME_STEP / 16)
TARGET_VELOCITY = 40  # [rad/s]
TARGET_STEERING_ANGLE = 10  # deg

data = AlamakData()
data_gt = AlamakDataGT()

# Controller
# -------------------------------------------------------------------------------------------------

# Skip initial simulation frames (filters out initial physics collisions and noise etc.)
timestamp = time_ns()

for i in range(3):
    robot.step(TIME_STEP)
    timestamp = time_ns()

step = 0

while robot.step(int(TIME_STEP)) != -1:
    control_data: ControlData | None = q_control.get(timeout=0)
    if control_data is not None:
        ackermann.angle = np.clip(control_data.steering_angle, -25, 25)
        print(ackermann.angle)
        motors.velocity = TARGET_VELOCITY
    else:
        motors.velocity = 0
    timestamp += TIME_STEP_NSEC

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
    steering_angle_estimate = estimate_steering_angle_deg(
        speed=speed_encoders, yaw_rate=angular_v_xyz[2]
    )
    data.steering_angle_estimate.append(steering_angle_estimate)

    # Datalink
    # ---------------------------------
    imu_data = IMU2Data()
    imu_data.timestamp = timestamp
    imu_data.accel_linear = a_xyz_imu
    imu_data.gyro = angular_v_xyz
    imu_data.mag = imu.get_compass()
    imu_data.rotation_euler_deg = rotation_xyz_imu
    imu_data.rotation_quaternion = imu.get_quaternion()
    encoder_data = EncoderData(timestamp=timestamp, position=rl_encoder.value_encoded, magnitude=20)
    jpg_image_data = JPGImageData(timestamp=timestamp, jpg=camera.get_jpeg())
    speedometer_data = SpeedometerData(timestamp=timestamp, dt=TIME_STEP_SEC, distance=0, speed=speed_encoders, encoder_data=encoder_data)
    sensor_fusion_data = SensorFusionData(
            timestamp=timestamp,
            last_timestamp=timestamp-TIME_STEP_NSEC,
            dt=TIME_STEP_SEC,
            avg_speed=speed_encoders,
            camera=jpg_image_data,
            speedometer=[speedometer_data],
            imu=[imu_data]
    )
    actuators_data = ActuatorsData(timestamp=timestamp, motor_power=1, steering_angle=steering_angle_estimate)
    control_data = ControlData(timestamp=timestamp, speed=speed_encoders, steering_angle=steering_angle_estimate)
    real_data = RealData(
        timestamp=timestamp,
        sensor_fusion=sensor_fusion_data,
        actuators=actuators_data,
        control=control_data
    )
    q_state.put(real_data, topic=b"data")

    if step == STEPS_TO_PLOT:
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        alamak_data_path = "data/alamak_data.pkl"
        alamak_data_gt_path = "data/alamak_data_gt.pkl"
        print("Saving data... ", end="")
        data.save(alamak_data_path)
        data_gt.save(alamak_data_gt_path)
        print("saved")
        # data_loaded: AlamakData = AlamakData.load(alamak_data_path)
        # data_gt_loaded: AlamakDataGT = AlamakDataGT.load(alamak_data_gt_path)
        # save_plots(data_loaded, data_gt_loaded)
        # robot.simulationReset()
        # robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
        break
    step += 1
