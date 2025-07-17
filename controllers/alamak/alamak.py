import cv2
import numpy as np
from time import time_ns

from controller import Robot, Camera
from alamaklib import AckermannSteering, Motors, Encoder, IMU

robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())
print(TIME_STEP)

ackermann = AckermannSteering(
    left_servo=robot.getDevice("fl_servo"), right_servo=robot.getDevice("fr_servo")
)
motors = Motors(left=robot.getDevice("rl_motor"), right=robot.getDevice("rr_motor"))
rl_encoder = Encoder(sensor=robot.getDevice("rl_encoder"), sampling_period_ms=TIME_STEP)
rr_encoder = Encoder(sensor=robot.getDevice("rr_encoder"), sampling_period_ms=TIME_STEP)
imu = IMU(
    gyro=robot.getDevice("imu_gyro"),
    accel=robot.getDevice("imu_accel"),
    compass=robot.getDevice("imu_compass"),
    sampling_period_ms=TIME_STEP,
)
camera: Camera = robot.getDevice("rpi_camera_v2")
camera.enable(TIME_STEP)
motors.velocity = 30

while robot.step(TIME_STEP) != -1:
    start = time_ns()
    ackermann.angle = 20
    argb = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()

    img = np.frombuffer(argb, dtype=np.uint8).reshape((height, width, 4))
    bgr_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    print((time_ns() - start) / 1e9, bgr_img.shape)