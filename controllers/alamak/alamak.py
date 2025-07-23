import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from controller import Supervisor
from alamaklib import (
    AckermannSteering,
    AlamakSupervisor,
    Motors,
    Encoder,
    IMU,
    CameraWrapper,
    transform_to_world_frame,
    decompose_acceleration,
)

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
    a = []
    a_xyz = []
    a_xyz_tangential = []
    a_xyz_normal = []
    v_xyz = []
    speed = []
    position_xyz = []
    rotation_xyz = []

@dataclass
class AlamakData:
    """
    Alamak vehicle Measured/Estimated data (obtained by sensors) 
    """
    timestamps = []
    a_xyz_imu = []
    a_xyz_tangential_imu = []
    a_xyz_normal_imu = []
    v_xyz_imu = []
    speed_imu = []
    position_xyz = []
    rotation_xyz = []
    wheel_steering_angles = []
    steering_angle_estimate = []

accels = []
accels_imu = []
accels_norm = []
accels_norm_imu = []
# Decomposed accelerations
tangential_acc = []
normal_acc = []
tangential_acc_imu = []
normal_acc_imu = []

velocities = []
velocities_imu = []

speeds = []
speeds_imu = []

positions = []
positions_imu = []

rotations = []
rotations_imu = []

wheel_steering_angles = []
steering_angles_estimated = []

STEPS_TO_PLOT = 1000
TARGET_VELOCITY = 20  # [rad/s]
TARGET_STEERING_ANGLE = 10  # deg

step = 0
motors.velocity = TARGET_VELOCITY
ackermann.angle = TARGET_STEERING_ANGLE

while robot.step(TIME_STEP) != -1:
    # Ground truth
    velocities.append(sup.velocity[:3])
    speeds.append(sup.speed)
    positions.append(sup.position)
    rotations.append(sup.rotation_euler_angles)
    wheel_steering_angles.append(
        (fl_servo_encoder.value_deg, fr_servo_encoder.value_deg)
    )
    a = (
        (velocities[step] - velocities[step - 1]) / TIME_STEP_SEC
        if step > 0
        else np.zeros(3)
    )
    accels.append(a)
    # a_norm = (speeds[step] - speeds[step - 1]) / TIME_STEP_SEC if step > 0 else 0.0
    a_norm = np.linalg.norm(a)
    accels_norm.append(a_norm)

    # Measured/Estimated
    accel = imu.get_accel() if step > 0 else [0.0, 0.0, 9.81]
    accel[2] -= 9.81  # Remove gravity component
    accel_world = transform_to_world_frame(accel, sup.rotation_euler_angles)
    accels_imu.append(accel_world)
    accels_norm_imu.append(np.linalg.norm(accel_world))

    # Estimate velocity by integrating acceleration
    if step == 0:
        velocities_imu.append(np.zeros(3))
    else:
        prev_vel = velocities_imu[step - 1]
        vel_estimated = prev_vel + np.array(accels_imu[step]) * TIME_STEP_SEC
        velocities_imu.append(vel_estimated)

    if step > 0:
        tang, norm = decompose_acceleration(velocities[step], accels[step])
        tangential_acc.append(tang)
        normal_acc.append(norm)

        tang_imu, norm_imu = decompose_acceleration(
            velocities_imu[step], accels_imu[step]
        )
        tangential_acc_imu.append(tang_imu)
        normal_acc_imu.append(norm_imu)
    else:
        tangential_acc.append(np.zeros(3))
        normal_acc.append(np.zeros(3))
        tangential_acc_imu.append(np.zeros(3))
        normal_acc_imu.append(np.zeros(3))

    if step == STEPS_TO_PLOT:
        accels = np.array(accels)
        accels_imu = np.array(accels_imu)
        velocities = np.array(velocities)
        velocities_imu = np.array(velocities_imu)
        positions = np.array(positions)
        rotations = np.array(rotations)
        wheel_steering_angles = np.array(wheel_steering_angles)
        steps = np.arange(len(positions))

        def position_plt(ax):
            ax.plot(steps, positions[:, 0], label="x", color="#d62728")
            ax.plot(steps, positions[:, 1], label="y", color="#2ca02c")
            ax.plot(steps, positions[:, 2], label="z", color="#1f77b4")
            ax.set_title("Position")
            ax.set_ylabel("Position [m]")
            ax.legend()
            ax.grid(True)

        def rotation_plt(ax):
            rot_arr_deg = np.rad2deg(rotations)
            ax.plot(steps, rot_arr_deg[:, 0], label="Roll", color="#d62728")
            ax.plot(steps, rot_arr_deg[:, 1], label="Pitch", color="#2ca02c")
            ax.plot(steps, rot_arr_deg[:, 2], label="Yaw", color="#1f77b4")
            ax.set_title("Orientation")
            ax.set_ylabel("Orientation [deg]")
            ax.set_ylim([-180, 180])
            ax.legend()
            ax.grid(True)

        def accel_plt(ax):
            ax.plot(steps, accels_norm, label="Ground Truth", color="#ff7f0e")
            ax.plot(
                steps, accels_norm_imu, "--", label="Accelerometer", color="#ff7f0e"
            )
            ax.set_title("Acceleration")
            ax.set_ylabel("Acceleration [m/s^2]")
            ax.legend()
            ax.grid(True)

        def accels_plt(ax):
            ax.plot(steps, accels[:, 0], label="x", color="#d62728")
            ax.plot(steps, accels[:, 1], label="y", color="#2ca02c")
            ax.plot(steps, accels[:, 2], label="z", color="#1f77b4")
            ax.plot(steps, accels_imu[:, 0], "--", label="x", color="#d62728")
            ax.plot(steps, accels_imu[:, 1], "--", label="y", color="#2ca02c")
            ax.plot(steps, accels_imu[:, 2], "--", label="z", color="#1f77b4")
            ax.set_title("Accelerations (x, y, z)")
            ax.set_ylabel("Acceleration  [m/s^2]")
            ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
            ax.grid(True)

        def speed_plt(ax):
            ax.plot(steps, speeds)
            ax.set_title("Speed")
            ax.set_ylabel("Speed [m/s]")
            ax.grid(True)

        def steering_plt(ax):
            ax.plot(steps, wheel_steering_angles[:, 0], label="Left Wheel")
            ax.plot(steps, wheel_steering_angles[:, 1], label="Right Wheel")
            ax.plot(
                steps,
                np.full_like(steps, ackermann.angle),
                "--",
                label="Target Angle (bicycle model)",
            )
            ax.set_title("Servo Angles")
            ax.set_ylabel("Angle [deg]")
            ax.legend()
            ax.grid(True)

        def velocities_plt(ax):
            ax.plot(steps, velocities[:, 0], label="x", color="#d62728")
            ax.plot(steps, velocities[:, 1], label="y", color="#2ca02c")
            ax.plot(steps, velocities[:, 2], label="z", color="#1f77b4")
            ax.plot(steps, velocities_imu[:, 0], "--", label="x", color="#d62728")
            ax.plot(steps, velocities_imu[:, 1], "--", label="y", color="#2ca02c")
            ax.plot(steps, velocities_imu[:, 2], "--", label="z", color="#1f77b4")
            ax.set_title("Velocities (x y z)")
            ax.set_ylabel("Velocity [m/s]")
            ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
            ax.grid(True)

        def decomposed_acceleration_plt(ax):
            # Magnitudes of tangential and normal accelerations
            tang_mag = [np.linalg.norm(a) for a in tangential_acc]
            norm_mag = [np.linalg.norm(a) for a in normal_acc]
            tang_mag_imu = [np.linalg.norm(a) for a in tangential_acc_imu]
            norm_mag_imu = [np.linalg.norm(a) for a in normal_acc_imu]

            ax.plot(steps, tang_mag, label="Tangential (speed change)", color="#d62728")
            ax.plot(steps, norm_mag, label="Normal (centripetal)", color="#2ca02c")
            ax.plot(
                steps, tang_mag_imu, "--", label="Tangential (IMU)", color="#d62728"
            )
            ax.plot(steps, norm_mag_imu, "--", label="Normal (IMU)", color="#2ca02c")

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

        fig, axs = plt.subplots(
            len(time_plts), 1, figsize=(10, len(time_plts) * 2), sharex=True
        )
        axs[-1].set_xlabel("Step")

        for i in range(len(time_plts)):
            time_plts[i](axs[i])

        # Time plots
        plt.savefig(
            "C:/Users/ihazu/Desktop/projects/toysim_webots/controllers/alamak/step_plots.png"
        )

        # Other plots
        plt.figure(figsize=(6, 6))
        plt.plot(positions[:, 0], positions[:, 1])
        plt.title("Position (x to y) [m]")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            "C:/Users/ihazu/Desktop/projects/toysim_webots/controllers/alamak/xy_plot.png"
        )

        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        break
        # robot.simulationReset()
        # robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

    step += 1
