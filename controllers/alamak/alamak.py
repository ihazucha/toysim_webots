from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from alamaklib import (
    IMU,
    AckermannSteering,
    AlamakSupervisor,
    CameraWrapper,
    Encoder,
    Motors,
    decompose_acceleration,
    transform_to_world_frame,
)
from controller import Supervisor

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


@dataclass
class AlamakDataGT:
    """
    Alamak vehicle Groud Truth data (obtained by a supevisor).
    """

    timestamps = []
    a_xyz = []
    a = []
    a_tangential = []
    a_normal = []
    v_xyz = []
    speed = []
    position_xyz = []
    rotation_xyz = []
    wheel_steering_angles = []


@dataclass
class AlamakData:
    """
    Alamak vehicle Measured/Estimated data (obtained by sensors)
    """

    timestamps = []
    a_xyz_imu = []
    a_imu = []
    a_tangential_imu = []
    a_normal_imu = []
    v_xyz_imu = []
    speed_imu = []
    position_xyz_imu = []
    rotation_xyz_imu = []
    steering_angle_estimate = []


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


def save_plots():
    data_gt.a_xyz = np.array(data_gt.a_xyz)
    data_gt.v_xyz = np.array(data_gt.v_xyz)
    data.a_xyz_imu = np.array(data.a_xyz_imu)
    data.v_xyz_imu = np.array(data.v_xyz_imu)
    data_gt.position_xyz = np.array(data_gt.position_xyz)
    data_gt.rotation_xyz = np.array(data_gt.rotation_xyz)
    data_gt.wheel_steering_angles = np.array(data_gt.wheel_steering_angles)
    steps = np.arange(len(data_gt.position_xyz))

    def position_plt(ax):
        ax.plot(steps, data_gt.position_xyz[:, 0], label="x", color="#d62728")
        ax.plot(steps, data_gt.position_xyz[:, 1], label="y", color="#2ca02c")
        ax.plot(steps, data_gt.position_xyz[:, 2], label="z", color="#1f77b4")
        ax.set_title("Position")
        ax.set_ylabel("Position [m]")
        ax.legend()
        ax.grid(True)

    def rotation_plt(ax):
        rot_arr_deg = np.rad2deg(data_gt.rotation_xyz)
        ax.plot(steps, rot_arr_deg[:, 0], label="Roll", color="#d62728")
        ax.plot(steps, rot_arr_deg[:, 1], label="Pitch", color="#2ca02c")
        ax.plot(steps, rot_arr_deg[:, 2], label="Yaw", color="#1f77b4")
        ax.set_title("Orientation")
        ax.set_ylabel("Orientation [deg]")
        ax.set_ylim([-180, 180])
        ax.legend()
        ax.grid(True)

    def accel_plt(ax):
        ax.plot(steps, data_gt.a, label="Ground Truth", color="#ff7f0e")
        ax.plot(steps, data.a_imu, "--", label="Accelerometer", color="#ff7f0e")
        ax.set_title("Acceleration")
        ax.set_ylabel("Acceleration [m/s^2]")
        ax.legend()
        ax.grid(True)

    def accels_plt(ax):
        ax.plot(steps, data_gt.a_xyz[:, 0], label="x", color="#d62728")
        ax.plot(steps, data_gt.a_xyz[:, 1], label="y", color="#2ca02c")
        ax.plot(steps, data_gt.a_xyz[:, 2], label="z", color="#1f77b4")
        ax.plot(steps, data.a_xyz_imu[:, 0], "--", label="x", color="#d62728")
        ax.plot(steps, data.a_xyz_imu[:, 1], "--", label="y", color="#2ca02c")
        ax.plot(steps, data.a_xyz_imu[:, 2], "--", label="z", color="#1f77b4")
        ax.set_title("Accelerations (x, y, z)")
        ax.set_ylabel("Acceleration  [m/s^2]")
        ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.grid(True)

    def speed_plt(ax):
        ax.plot(steps, data_gt.speed)
        ax.set_title("Speed")
        ax.set_ylabel("Speed [m/s]")
        ax.grid(True)

    def steering_plt(ax):
        ax.plot(steps, data_gt.wheel_steering_angles[:, 0], label="Left Wheel")
        ax.plot(steps, data_gt.wheel_steering_angles[:, 1], label="Right Wheel")
        ax.axhline(y=ackermann.angle, linestyle="--", label="Target Steering Angle (bicycle model)")
        ax.set_title("Servo Angles")
        ax.set_ylabel("Angle [deg]")
        ax.legend()
        ax.grid(True)

    def velocities_plt(ax):
        ax.plot(steps, data_gt.v_xyz[:, 0], label="x", color="#d62728")
        ax.plot(steps, data_gt.v_xyz[:, 1], label="y", color="#2ca02c")
        ax.plot(steps, data_gt.v_xyz[:, 2], label="z", color="#1f77b4")
        ax.plot(steps, data.v_xyz_imu[:, 0], "--", label="x", color="#d62728")
        ax.plot(steps, data.v_xyz_imu[:, 1], "--", label="y", color="#2ca02c")
        ax.plot(steps, data.v_xyz_imu[:, 2], "--", label="z", color="#1f77b4")
        ax.set_title("Velocities (x y z)")
        ax.set_ylabel("Velocity [m/s]")
        ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.grid(True)

    def decomposed_acceleration_plt(ax):
        ax.plot(steps, data_gt.a_tangential, label="Tangential (speed change)", color="#d62728")
        ax.plot(steps, data_gt.a_normal, label="Normal (centripetal)", color="#2ca02c")
        ax.plot(steps, data.a_tangential_imu, "--", label="Tangential (IMU)", color="#d62728")
        ax.plot(steps, data.a_normal_imu, "--", label="Normal (IMU)", color="#2ca02c")
        ax.set_title("Decomposed Acceleration")
        ax.set_ylabel("Acceleration [m/s²]")
        ax.legend()
        ax.grid(True)

    time_plts = [
        accel_plt,
        speed_plt,
        accels_plt,
        velocities_plt,
        decomposed_acceleration_plt,
        position_plt,
        steering_plt,
        rotation_plt,
    ]

    plt.figure()

    fig, axs = plt.subplots(len(time_plts), 1, figsize=(10, len(time_plts) * 2), sharex=True)
    axs[-1].set_xlabel("Step")

    for i in range(len(time_plts)):
        time_plts[i](axs[i])

    # Time plots
    plt.savefig("C:/Users/ihazu/Desktop/projects/toysim_webots/controllers/alamak/step_plots.png")

    # Other plots
    plt.figure(figsize=(6, 6))
    plt.plot(data_gt.position_xyz[:, 0], data_gt.position_xyz[:, 1])
    plt.title("Position (x to y) [m]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("C:/Users/ihazu/Desktop/projects/toysim_webots/controllers/alamak/xy_plot.png")


while robot.step(TIME_STEP) != -1:
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
    data_gt.wheel_steering_angles.append((fl_servo_encoder.value_deg, fr_servo_encoder.value_deg))

    # Estimated
    a_xyz_imu_local_frame = imu.get_linear_acceleration()
    a_xyz_imu = transform_to_world_frame(a_xyz_imu_local_frame, sup.rotation_euler_angles)
    data.a_xyz_imu.append(a_xyz_imu)
    data.a_imu.append(np.linalg.norm(a_xyz_imu))
    v_xyz_imu = (
        np.zeros(3)
        if step == 0
        else data.v_xyz_imu[step - 1] + data.a_xyz_imu[step] * TIME_STEP_SEC
    )
    data.v_xyz_imu.append(v_xyz_imu)
    a_tang_imu, a_norm_imu = decompose_acceleration(v_xyz_imu, a_xyz_imu)
    data.a_tangential_imu.append(a_tang_imu)
    data.a_normal_imu.append(a_norm_imu)

    if step == STEPS_TO_PLOT:
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        save_plots()
        break
        # robot.simulationReset()
        # robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

    step += 1
