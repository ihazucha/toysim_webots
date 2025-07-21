import numpy as np
import matplotlib.pyplot as plt

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
# TIME_STEP = int(robot.getBasicTimeStep())
TIME_STEP = 16

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

motors.velocity = 30
ackermann.angle = 10

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

        # Plot all "step" dependent plots in one figure
        fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

        # Position vs Step
        axs[0].plot(step_arr, pos_arr[:, 0], label="x")
        axs[0].plot(step_arr, pos_arr[:, 1], label="y")
        axs[0].plot(step_arr, pos_arr[:, 2], label="z")
        axs[0].set_title("Position")
        axs[0].set_ylabel("Position [m]")
        axs[0].legend()
        axs[0].grid(True)

        # Orientation vs Step
        ori_arr_deg = np.rad2deg(ori_arr)
        axs[1].plot(step_arr, ori_arr_deg[:, 0], label="Roll")
        axs[1].plot(step_arr, ori_arr_deg[:, 1], label="Pitch")
        axs[1].plot(step_arr, ori_arr_deg[:, 2], label="Yaw")
        axs[1].set_title("Orientation")
        axs[1].set_ylabel("Orientation [rad]")
        axs[1].set_ylim([-180, 180])
        axs[1].grid(True)

        # Acceleration vs Step
        axs[2].plot(step_arr, accelerations)
        axs[2].set_title("Acceleration")
        axs[2].set_ylabel("Acceleration [m/s²]")
        axs[2].grid(True)

        # Velocity vs Step  
        axs[3].plot(step_arr, speeds)
        axs[3].set_title("Velocity")
        axs[3].set_ylabel("Velocity [m/s]")
        axs[3].grid(True)

        # Servo Encoders vs Step
        axs[4].plot(step_arr, servo_arr[:, 0], label="Left Wheel")
        axs[4].plot(step_arr, servo_arr[:, 1], label="Right Wheel")
        axs[4].plot(step_arr, np.full_like(step_arr, ackermann.angle), '--', label="Target Angle (bicycle model)")
        axs[4].set_title("Servo Angles")
        axs[4].set_xlabel("Step")
        axs[4].set_ylabel("Angle [deg]")
        axs[4].legend()
        axs[4].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig('C:/Users/ihazu/Desktop/projects/toysim_webots/controllers/alamak/step_plots.png')

        # Separate position plot (x vs y)
        plt.figure(figsize=(6, 6))
        plt.plot(pos_arr[:, 0], pos_arr[:, 1])
        plt.title("Position (x to y) [m]")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('C:/Users/ihazu/Desktop/projects/toysim_webots/controllers/alamak/xy_plot.png')
        break
        # robot.simulationReset()
        # robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
