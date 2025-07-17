import math

from controller import (
    Accelerometer,
    InertialUnit,
    Compass,
    PositionSensor,
    Motor,
)

class AlamakParams:
    WHEELBASE = 0.185
    TRACK = 0.155


class AckermannSteering:
    def __init__(
        self,
        left_servo: Motor,
        right_servo: Motor,
        wheelbase: float = AlamakParams.WHEELBASE,
        track: float = AlamakParams.TRACK,
    ):
        self.left_servo = left_servo
        self.right_servo = right_servo
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

        self.left_servo.setPosition(left_angle)
        self.right_servo.setPosition(right_angle)


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
    RESOLUTION = 4096

    def __init__(self, sensor: PositionSensor, sampling_period_ms: int = 16):
        self.sensor = sensor
        self.sensor.enable(sampling_period_ms)

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
    def __init__(
        self,
        gyro: InertialUnit,
        accel: Accelerometer,
        compass: Compass,
        sampling_period_ms: int = 16,
    ):
        self.gyro = gyro
        self.accel = accel
        self.compass = compass

        self.gyro.enable(sampling_period_ms)
        self.accel.enable(sampling_period_ms)
        self.compass.enable(sampling_period_ms)

    def get_gyro(self):
        return self.gyro.getRollPitchYaw()

    def get_accel(self):
        return self.accel.getValues()

    def get_compass(self):
        return self.compass.getValues()
