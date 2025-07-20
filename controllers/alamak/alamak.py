import numpy as np
from time import time_ns
import matplotlib.pyplot as plt
from threading import Thread

from controller import Supervisor
from alamaklib import AckermannSteering, AlamakSupervisor, Motors, Encoder, IMU, CameraWrapper
from dataclasses import dataclass

@dataclass
class VehicleState:
    acceleration = 0.0
    velocity = 0.0
    position = np.zeros(3)
    orientation = np.zeros(3)
    target_steering_angle = 0.0
    left_wheel_steering_angle = 0.0
    right_wheel_steering_angle = 0.0
    front_wheel_speeds = np.zeros(2)
    rear_wheel_speeds = np.zeros(2)


robot = Supervisor()
TIME_STEP = int(robot.getBasicTimeStep())

ackermann = AckermannSteering(
    left_servo=robot.getDevice("fl_servo"), right_servo=robot.getDevice("fr_servo")
)
motors = Motors(left=robot.getDevice("rl_motor"), right=robot.getDevice("rr_motor"))
rl_encoder = Encoder(sensor=robot.getDevice("rl_encoder"), sampling_period_ms=TIME_STEP)
rr_encoder = Encoder(sensor=robot.getDevice("rr_encoder"), sampling_period_ms=TIME_STEP)
fl_servo_encoder = Encoder(sensor=robot.getDevice("fl_servo_encoder"), sampling_period_ms=TIME_STEP)
fr_servo_encoder = Encoder(sensor=robot.getDevice("fr_servo_encoder"), sampling_period_ms=TIME_STEP)

imu = IMU(
    gyro=robot.getDevice("imu_gyro"),
    accel=robot.getDevice("imu_accel"),
    compass=robot.getDevice("imu_compass"),
    sampling_period_ms=TIME_STEP,
)

camera = CameraWrapper(
    camera=robot.getDevice("rpi_camera_v2"), sampling_period_ms=TIME_STEP
)
motors.velocity = 30
ackermann.angle = 10

alamak_supervisor = AlamakSupervisor(robot)

# Scenarios
# 1. Straight line
# 2. Circle
# 3. Figure 8

positions = []
orientations = []
speeds = []
accelerations = []
servo_encoders = []

steps_to_plot = 500


while robot.step(TIME_STEP) != -1:

    positions.append(alamak_supervisor.position)
    orientations.append(alamak_supervisor.rotation)
    speeds.append(alamak_supervisor.speed)
    servo_encoders.append((fl_servo_encoder.value_deg, fr_servo_encoder.value_deg))

    if len(positions) % steps_to_plot == 0:
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        accelerations.clear()
        for i in range(len(speeds)):
            if i == 0:
                accelerations.append(0.0)
            else:
                acceleration = (speeds[i] - speeds[i - 1]) / (TIME_STEP / 1000.0)
                accelerations.append(acceleration)
        
        plt.figure(figsize=(12, 8))

        # Prepare arrays
        pos_arr = np.array(positions)
        ori_arr = np.array(orientations)
        servo_arr = np.array(servo_encoders)
        step_arr = np.arange(len(positions))

        fig, axs = plt.subplots(5, 2, figsize=(14, 18))

        # Left column: step on x-axis
        axs[0, 0].plot(step_arr, pos_arr[:, 0], label="x")
        axs[0, 0].plot(step_arr, pos_arr[:, 1], label="y")
        axs[0, 0].set_title("Position vs Step")
        axs[0, 0].set_xlabel("Step")
        axs[0, 0].set_ylabel("Position (m)")
        axs[0, 0].legend()

        axs[1, 0].plot(step_arr, ori_arr)
        axs[1, 0].set_title("Orientation vs Step")
        axs[1, 0].set_xlabel("Step")
        axs[1, 0].set_ylabel("Orientation (radians)")
        axs[1, 0].set_ylim([-np.pi, np.pi])

        axs[2, 0].plot(step_arr, accelerations)
        axs[2, 0].set_title("Acceleration vs Step")
        axs[2, 0].set_xlabel("Step")
        axs[2, 0].set_ylabel("Acceleration (m/s²)")

        axs[3, 0].plot(step_arr, speeds)
        axs[3, 0].set_title("Velocity vs Step")
        axs[3, 0].set_xlabel("Step")
        axs[3, 0].set_ylabel("Velocity (m/s)")

        axs[4, 0].plot(step_arr, servo_arr[:, 0], label="FL Servo Encoder")
        axs[4, 0].plot(step_arr, servo_arr[:, 1], label="FR Servo Encoder")
        axs[4, 0].plot(step_arr, np.full_like(step_arr, ackermann.angle), '--', label="Constant 10 deg (rad)")
        axs[4, 0].set_title("Servo Encoders vs Step")
        axs[4, 0].set_xlabel("Step")
        axs[4, 0].set_ylabel("Servo Encoder / Constant (rad)")
        axs[4, 0].legend()

        # Right column: position plots
        axs[0, 1].plot(pos_arr[:, 0], pos_arr[:, 1])
        axs[0, 1].set_title("Position (x to y) [m]")
        axs[0, 1].set_xlabel("x")
        axs[0, 1].set_ylabel("y")
        axs[0, 1].axis('equal')

        # Hide unused subplots in right column
        for i in range(1, 5):
            fig.delaxes(axs[i, 1])

        plt.tight_layout()
        plt.savefig('C:/Users/ihazu/Desktop/projects/toysim_webots/controllers/alamak/plot.png')
        break
        # robot.simulationReset()
        # robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
