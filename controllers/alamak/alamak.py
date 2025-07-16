from controller import (
    Robot,
    Accelerometer,
    InertialUnit,
    Compass,
    Camera,
    PositionSensor,
    Motor,
)
import math
import cv2
import numpy as np
from time import time_ns

"""
1. Car measurements
    - track = 0.155 [m]
    - wheelbase = 0.185 [m]
    - wheel_radius = 0.032 [m]
    - max_steer_angle = 20 [deg]
    - vehicle_len = 0.265 [m]
    - vehicle_width = 0.185 [m]
2. Figure out vehicle physics parameters:
    - Weight
    - CoM
    - Estimate inertia matrix
3. Figure out motor paramters
4. Implement ackermann steering geometry
5. Implement sensors
    - Rotary encoders
    - IMU
    - Camera
6. Read all to data
7. Setup communication by importing data-link

"""

TIME_STEP = 16

WHEELBASE = 0.185
TRACK = 0.155

robot = Robot()

PHYS_TIME_STEP = robot.getBasicTimeStep()


class AckermannSteering:
    def __init__(
        self,
        left_wheel: Motor,
        right_wheel: Motor,
        wheelbase: float = WHEELBASE,
        track: float = TRACK,
    ):
        self.left_wheel = left_wheel
        self.right_wheel = right_wheel
        self.wheelbase = wheelbase
        self.track = track
        self.track_half = self.track / 2
        self._angle = 0

    @property
    def angle(self) -> float:
        return self._angle

    @angle.setter
    def angle(self, angle_deg: float):
        assert -20 <= angle_deg <= 20, "Value out of range <-20, 20>"
        self._set_wheel_angles(angle_deg)
        self._angle = angle_deg

    def _set_wheel_angles(self, angle_deg: float):
        angle = math.radians(angle_deg)
        R = self.wheelbase / math.tan(angle)
        if angle > 0:  # left
            left_angle = math.atan(self.wheelbase / (R - self.track_half))
            right_angle = math.atan(self.wheelbase / (R + self.track_half))
        else:
            left_angle = math.atan(self.wheelbase / (R + self.track_half))
            right_angle = math.atan(self.wheelbase / (R - self.track_half))

        self.left_wheel.setPosition(left_angle)
        self.right_wheel.setPosition(right_angle)


class Motors:
    def __init__(self, left: Motor, right: Motor):
        self.left = left
        self.right = right
        self._velocity = 0

    @property
    def velocity(self) -> float:
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: float):
        p, v = (
            (float("inf"), velocity) if velocity >= 0.0 else (float("-inf"), -velocity)
        )
        for m in (self.left, self.right):
            m.setPosition(p)
            m.setVelocity(v)
        self._velocity = velocity


class Encoder:
    # According to https://www.mouser.com/pdfdocs/AMS_AS5600_Datasheet_EN.PDF
    # Sampling rate (meant period according to provided units) is 150 microseconds
    # ! PositionSensor.enable() only accepts int value in miliseconds.
    T_SAMPLE = int(PHYS_TIME_STEP)
    RESOLUTION = 4096

    def __init__(self, sensor: PositionSensor):
        self.sensor = sensor
        self.sensor.enable(self.__class__.T_SAMPLE)

    def __str__(self):
        return f"(rad, deg, enc) = ({self.value_rad:.2f}, {self.value_deg:6.2f}, {int(self.value_encoded):4d})"

    @property
    def value_rad(self) -> float:
        return self.sensor.getValue()

    @property
    def value_deg(self) -> float:
        return math.degrees(self.sensor.getValue()) % 360

    @property
    def value_encoded(self) -> int:
        return int((self.value_deg) / 360 * self.__class__.RESOLUTION)


class IMU:
    T_SAMPLE = int(PHYS_TIME_STEP)

    def __init__(self, gyro: InertialUnit, accel: Accelerometer, compass: Compass):
        self.gyro = gyro
        self.accel = accel
        self.compass = compass

        self.gyro.enable(self.__class__.T_SAMPLE)
        self.accel.enable(self.__class__.T_SAMPLE)
        self.compass.enable(self.__class__.T_SAMPLE)

    def get_gyro(self):
        return self.gyro.getRollPitchYaw()

    def get_accel(self):
        return self.accel.getValues()

    def get_compass(self):
        return self.compass.getValues()



ackermann = AckermannSteering(
    left_wheel=robot.getDevice("alamak_fl_steer"),
    right_wheel=robot.getDevice("alamak_fr_steer"),
)

motors = Motors(
    left=robot.getDevice("alamak_rl_motor"),
    right=robot.getDevice("alamak_rr_motor"),
)

rl_encoder = Encoder(sensor=robot.getDevice("alamak_rl_position_sensor"))
rr_encoder = Encoder(sensor=robot.getDevice("alamak_rr_position_sensor"))

imu = IMU(
    gyro=robot.getDevice("alamak_gyro"),
    accel=robot.getDevice("alamak_accel"),
    compass=robot.getDevice("alamak_compass"),
)
camera: Camera = robot.getDevice("alamak_camera")
camera.enable(int(PHYS_TIME_STEP))

motors.velocity = 10

time = 0.0

while robot.step(TIME_STEP) != -1:
    start = time_ns()
    time += TIME_STEP / 1000.0
    direction = math.sin(time) / 2.8647889756541160438399077407053
    ackermann.angle = 10
    argb = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()

    # Convert Webots ARGB image to numpy array
    img = np.frombuffer(argb, dtype=np.uint8).reshape((height, width, 4))
    bgr_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # success, jpg_bytes = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    print((time_ns() - start) / 1e9, len(bgr_img))
