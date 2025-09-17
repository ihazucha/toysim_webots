import matplotlib.pyplot as plt
import numpy as np
from data import AlamakData, AlamakDataGT


class Colors:
    RED = "#d62728"
    GREEN = "#2ca02c"
    BLUE = "#1f77b4"
    ORANGE = "#ff7f0e"
    LIME = "#bcbd22"
    CYAN = "#17becf"
    PURPLE = "#9467bd"
    BROWN = "#8c564b"
    PINK = "#e377c2"


def save_plots(data: AlamakData, data_gt: AlamakDataGT):
    steps = np.arange(len(data_gt.a_xyz))
    a_xyz = np.array(data_gt.a_xyz)
    v_xyz = np.array(data_gt.v_xyz)
    position_xyz = np.array(data_gt.position_xyz)
    rotation_xyz = np.array(data_gt.rotation_xyz)
    rotation_xyz_imu = np.array(data.rotation_xyz_imu)
    a_xyz_imu = np.array(data.a_xyz_imu)
    wheel_steering_angles = np.array(data_gt.wheel_steering_angles)
    v_xyz_imu = np.array(data.v_xyz_imu)
    angular_v_xyz_imu = np.array(data.angular_v_xyz_imu)
    speed_wheels = np.array(data.speed_wheels)

    def position_plt(ax):
        ax.plot(steps, position_xyz[:, 0], label="x", color="#d62728")
        ax.plot(steps, position_xyz[:, 1], label="y", color="#2ca02c")
        ax.plot(steps, position_xyz[:, 2], label="z", color="#1f77b4")
        ax.set_title("Position")
        ax.set_ylabel("Position [m]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def rotation_plt(ax):
        rotation_xyz_deg = np.rad2deg(rotation_xyz)
        rotation_xyz_imu_deg = np.rad2deg(rotation_xyz_imu)
        ax.plot(steps, rotation_xyz_deg[:, 0], label="x", color="#d62728")
        ax.plot(steps, rotation_xyz_deg[:, 1], label="y", color="#2ca02c")
        ax.plot(steps, rotation_xyz_deg[:, 2], label="z", color="#1f77b4")
        ax.plot(steps, rotation_xyz_imu_deg[:, 0], "--", label="x (IMU)", color="#d62728")
        ax.plot(steps, rotation_xyz_imu_deg[:, 1], "--", label="y (IMU)", color="#2ca02c")
        ax.plot(steps, rotation_xyz_imu_deg[:, 2], "--", label="z (IMU)", color="#1f77b4")
        ax.set_title("Orientation")
        ax.set_ylabel("Orientation [deg]")
        ax.set_ylim([-180, 180])
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def accels_plt(ax):
        ax.plot(steps, a_xyz[:, 0], label="x", color="#d62728")
        ax.plot(steps, a_xyz[:, 1], label="y", color="#2ca02c")
        ax.plot(steps, a_xyz[:, 2], label="z", color="#1f77b4")
        ax.plot(steps, a_xyz_imu[:, 0], "--", label="x (IMU)", color="#d62728")
        ax.plot(steps, a_xyz_imu[:, 1], "--", label="y (IMU)", color="#2ca02c")
        ax.plot(steps, a_xyz_imu[:, 2], "--", label="z (IMU)", color="#1f77b4")
        ax.set_title("Accelerations (x, y, z)")
        ax.set_ylabel("Acceleration  [m/s^2]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def speed_plt(ax):
        ax.plot(steps, data_gt.speed, label="Ground Truth", color=Colors.ORANGE)
        ax.plot(steps, data.speed_encoders, "--", label="Encoder", color=Colors.ORANGE)
        ax.plot(steps, data.speed_imu, ":", label="IMU", color=Colors.ORANGE)
        ax.set_title("Speed")
        ax.set_ylabel("Speed [m/s]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def speed_wheels_plt(ax):
        ax.plot(steps, speed_wheels[:, 0], label="Rear left", color=Colors.LIME)
        ax.plot(steps, speed_wheels[:, 1], label="Rear right", color=Colors.CYAN)
        ax.set_title("Wheels speed")
        ax.set_ylabel("Speed [m/s]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def steering_plt(ax):
        ax.plot(steps, wheel_steering_angles[:, 0], label="Left Wheel", color=Colors.PINK)
        ax.plot(steps, wheel_steering_angles[:, 1], label="Right Wheel", color=Colors.PURPLE)
        ax.plot(
            steps,
            data.target_steering_angle,
            linestyle="--",
            label="Target Steering Angle",
            color=Colors.BROWN,
        )
        ax.plot(
            steps, data.steering_angle_estimate, label="Steering angle estimate", color=Colors.BROWN
        )
        ax.set_title("Servo Angles")
        ax.set_ylabel("Angle [deg]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend(loc='center right')
        ax.grid(True)

    def velocities_plt(ax):
        ax.plot(steps, v_xyz[:, 0], label="x", color="#d62728")
        ax.plot(steps, v_xyz[:, 1], label="y", color="#2ca02c")
        ax.plot(steps, v_xyz[:, 2], label="z", color="#1f77b4")
        ax.plot(steps, v_xyz_imu[:, 0], "--", label="x (IMU)", color="#d62728")
        ax.plot(steps, v_xyz_imu[:, 1], "--", label="y (IMU)", color="#2ca02c")
        ax.plot(steps, v_xyz_imu[:, 2], "--", label="z (IMU)", color="#1f77b4")
        ax.set_title("Velocities (x, y, z)")
        ax.set_ylabel("Velocity [m/s]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def angular_velocities_plt(ax):
        ax.plot(steps, angular_v_xyz_imu[:, 0], label="x", color="#d62728")
        ax.plot(steps, angular_v_xyz_imu[:, 1], label="y", color="#2ca02c")
        ax.plot(steps, angular_v_xyz_imu[:, 2], label="z", color="#1f77b4")
        ax.set_title("Angular Velocities (x, y, z)")
        ax.set_ylabel("Angular vel   [rad/s]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    def decomposed_acceleration_plt(ax):
        ax.plot(steps, data_gt.a_tangential, label="Tangential (speed change)", color="#d62728")
        ax.plot(steps, data_gt.a_normal, label="Normal (centripetal)", color="#2ca02c")
        ax.plot(steps, data.a_tangential_imu, "--", label="Tangential (IMU)", color="#d62728")
        ax.plot(steps, data.a_normal_imu, "--", label="Normal (IMU)", color="#2ca02c")
        ax.set_title("Decomposed Acceleration")
        ax.set_ylabel("Acceleration [m/s²]")
        # ax.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
        ax.legend()
        ax.grid(True)

    time_plts = [
        # accels_plt,
        # decomposed_acceleration_plt,
        velocities_plt,
        speed_plt,
        speed_wheels_plt,
        angular_velocities_plt,
        steering_plt,
        rotation_plt,
        position_plt,
    ]

    plt.figure()

    fig, axs = plt.subplots(len(time_plts), 1, figsize=(10, len(time_plts) * 2), sharex=True)
    axs[-1].set_xlabel("Step")

    for i in range(len(time_plts)):
        time_plts[i](axs[i])

    # Time plots
    plt.savefig("plots/step_plots.png")

    # Other plots
    plt.figure(figsize=(6, 6))
    plt.plot(position_xyz[:, 0], position_xyz[:, 1])
    plt.title("Position (x to y) [m]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/xy_plot.png")


if __name__ == "__main__":
    data: AlamakData = AlamakData.load("data/alamak_data.pkl")
    data_gt: AlamakDataGT = AlamakDataGT.load("data/alamak_data_gt.pkl")
    save_plots(data, data_gt)
