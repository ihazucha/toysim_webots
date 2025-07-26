from dataclasses import dataclass, field
import pickle


@dataclass
class AlamakParams:
    WHEELBASE = 0.185
    TRACK = 0.155
    WHEEL_RADIUS = 0.032

@dataclass
class DataSerializable:
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


@dataclass
class AlamakDataGT(DataSerializable):
    """
    Alamak vehicle Ground Truth data (obtained by a supervisor).
    """

    timestamps: list = field(default_factory=list)
    a_xyz: list = field(default_factory=list)
    a: list = field(default_factory=list)
    a_tangential: list = field(default_factory=list)
    a_normal: list = field(default_factory=list)
    v_xyz: list = field(default_factory=list)
    speed: list = field(default_factory=list)
    position_xyz: list = field(default_factory=list)
    rotation_xyz: list = field(default_factory=list)
    wheel_steering_angles: list = field(default_factory=list)


@dataclass
class AlamakData(DataSerializable):
    """
    Alamak vehicle Measured/Estimated data (obtained by sensors)
    """

    timestamps: list = field(default_factory=list)
    # Inputs
    target_steering_angle: list = field(default_factory=list)
    target_speed: list = field(default_factory=list)
    # Measurements
    a_xyz_imu: list = field(default_factory=list)
    a_imu: list = field(default_factory=list)
    a_tangential_imu: list = field(default_factory=list)
    a_normal_imu: list = field(default_factory=list)
    v_xyz_imu: list = field(default_factory=list)
    speed_imu: list = field(default_factory=list)
    speed_encoders: list = field(default_factory=list)
    speed_wheels: list = field(default_factory=list)
    position_xyz_imu: list = field(default_factory=list)
    rotation_xyz_imu: list = field(default_factory=list)
    angular_v_xyz_imu: list = field(default_factory=list)
    steering_angle_estimate: list = field(default_factory=list)