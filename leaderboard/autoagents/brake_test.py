import socket
import json
import math
import numpy as np
import open3d as o3d
import threading
import struct
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from carla import VehicleControl

# UDP_IP = "10.16.78.229"
# TCP_IP = "10.16.78.229"
# TCP_IP = "10.16.88.213"
# UDP_IP = "10.16.88.213"
TCP_IP = "192.168.1.101"
UDP_IP = "192.168.1.101"
UDP_CONTROL_IP = "192.168.1.1"

UDP_PORT_GNSS_IMU = 12345
UDP_PORT_CONTROL = 23456
TCP_PORT_LIDAR = 5005

sock_gnss_imu = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def deg_heading_from_compass(compass_radian):
    return math.degrees(compass_radian)


def downsample_pointcloud(xyz, voxel_size=0.3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    downsampled = pcd.voxel_down_sample(voxel_size)
    return np.asarray(downsampled.points)

# 新增：定义镜像函数
def mirror_point_cloud(xyz):
    """
    对 LiDAR 点云数据沿中轴（x = 0）进行镜像处理。

    参数:
    xyz (numpy.ndarray): 包含点云数据的二维数组，每一行是一个点的 [x, y, z] 坐标。

    返回:
    numpy.ndarray: 镜像后的点云数据。
    """
    mirrored_xyz = xyz.copy()
    mirrored_xyz[:, 0] = -mirrored_xyz[:, 0]
    return mirrored_xyz

class MyAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        self.current_heading = 0.0
        self.prev_timestamp = None
        self.heading_deg = 0.0  # 新增：用于存储相对于正东方向的夹角

        self.control_command = {"steer": 0.0, "throttle": 0.0, "brake": 1.0}
        self.control_lock = threading.Lock()
        self.gps_file = open("gps_path.txt", "w")

        self.filter_window_size = 5
        self.gps_x_buffer = []
        self.gps_y_buffer = []

        # LiDAR点缓冲设置
        self.lidar_buffer = []
        self.lidar_buffer_lock = threading.Lock()
        self.lidar_batch_size = 50000
        self.biaozhi = 1
        threading.Thread(target=self.receive_control_loop, daemon=True).start()
        self.tcp_lidar_socket = None
        threading.Thread(target=self.connect_tcp_lidar, daemon=True).start()

    def sensors(self):
        return [
            {
                "type": "sensor.other.gnss", "id": "GPS",
                "x": 0, "y": 0, "z": 1.60,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0
            },
            {
                "type": "sensor.other.gnss", "id": "GPS1",
                "x": 1, "y": 0, "z": 1.60,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0
            },
            {
                "type": "sensor.other.imu", "id": "IMU",
                "x": 0, "y": 0, "z": 1.60,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0
            },
            {
                "type": "sensor.lidar.ray_cast", "id": "LIDAR",
                "x": 0, "y": 0.0, "z": 1.8,
                "roll": 0, "pitch": 0, "yaw": -90,
                "range": 5, "rotation_frequency": 10, "channels": 16,
                "upper_fov": 10, "lower_fov": -25, "points_per_second": 50000
            }
        ]

    def connect_tcp_lidar(self):
        while self.tcp_lidar_socket is None:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((TCP_IP, TCP_PORT_LIDAR))
                self.tcp_lidar_socket = sock
                print(f"[LIDAR TCP] 已连接到 {TCP_IP}:{TCP_PORT_LIDAR}")
            except Exception as e:
                print(f"[LIDAR TCP] 连接失败，重试中... ({e})")
                import time
                time.sleep(1)

    def send_lidar_buffer(self):
        with self.lidar_buffer_lock:
            if not self.lidar_buffer:
                return
            all_points = np.vstack(self.lidar_buffer)
            self.lidar_buffer.clear()

        # 降采样处理
        downsampled_points = downsample_pointcloud(all_points)

        scaled_xyz = (downsampled_points * 100).astype(np.int16)
        data_bytes = scaled_xyz.tobytes()
        header = struct.pack("!I", len(data_bytes))

        try:
            self.tcp_lidar_socket.sendall(header + data_bytes)
            print(f"[LIDAR] 已发送 {scaled_xyz.shape[0]} 个点（带帧头）")
            del scaled_xyz
        except Exception as e:
            print(f"[LIDAR 发送错误] {e}")
            self.tcp_lidar_socket = None
            threading.Thread(target=self.connect_tcp_lidar, daemon=True).start()

    def receive_control_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_CONTROL_IP, UDP_PORT_CONTROL))
        print(f"[控制接收] 正在监听 UDP {UDP_PORT_CONTROL}...")

        while True:
            try:
                data, _ = sock.recvfrom(1024)
                msg = json.loads(data.decode('utf-8'))
                if msg.get("id") == "control":
                    with self.control_lock:
                        self.control_command["steer"] = float(msg.get("steer", 0.0))
                        self.control_command["throttle"] = float(msg.get("throttle", 0.0))
                        self.control_command["brake"] = float(msg.get("brake", 0.0))
                        print(f"[控制指令] Steer: {self.control_command['steer']:.2f}, Throttle: {self.control_command['throttle']:.2f}, Brake: {self.control_command['brake']:.2f}")
            except Exception as e:
                print(f"[控制接收错误] {e}")

    def run_step(self, input_data, timestamp):
        if 'GPS' in input_data and 'GPS1' in input_data:
            gps_data = input_data['GPS']
            x1, y1, z1 = gps_data[1]

            gps1_data = input_data['GPS1']
            x2, y2, z2 = gps1_data[1]

            # 计算两点连线相对于正东方向的夹角
            dx = x2 - x1
            dy = y2 - y1
            angle = math.atan2(dy, dx)
            self.heading_deg = math.degrees(angle)
            if dx < 0:
                self.heading_deg += 180
            elif dx > 0 and dy < 0:
                self.heading_deg += 360
            # self.heading_deg += 90
            print(f"[GPS夹角] 相对于正东方向的夹角: {self.heading_deg:.2f} 度")

        if 'GPS' in input_data:
            gps_data = input_data['GPS']
            x, y, z = gps_data[1]

            # 新增：当GPS的x坐标大于100时进行刹车
            # if x > -0.03198133615231313:
            if x > 1:

                with self.control_lock:
                    self.control_command["brake"] = 100.0
                    self.biaozhi = 0

            # self.gps_x_buffer.append(x)
            # self.gps_y_buffer.append(y)
            # if len(self.gps_x_buffer) > self.filter_window_size:
            #     self.gps_x_buffer.pop(0)
            #     self.gps_y_buffer.pop(0)
            # filtered_x = np.mean(self.gps_x_buffer)
            # filtered_y = np.mean(self.gps_y_buffer)

            msg = {"id": "gnss", "x": x, "y": y, "z": z}
            print(f"[GNSS] x: {x:.12f}, y: {y:.12f}")
            sock_gnss_imu.sendto(json.dumps(msg).encode('utf-8'), (UDP_IP, UDP_PORT_GNSS_IMU))
            self.gps_file.write(f"{x} {y}\n")

        if 'IMU' in input_data:
            imu_data = input_data['IMU']
            imu_values = imu_data[1]
            if len(imu_values) in [6, 7]:
                ax, ay, az, gx, gy, gz = imu_values[:6]
                # dt = timestamp - self.prev_timestamp if self.prev_timestamp is not None else 0.0
                # self.current_heading += gz * dt
                # 这里可以选择使用GPS计算的heading_deg，或者将两者融合
                # 暂时注释掉原有的heading_deg计算
                # heading_deg = deg_heading_from_compass(self.current_heading)
                # self.heading_deg += 90
                # if 0 <= self.heading_deg < 90:
                #     self.heading_deg = 90 - self.heading_deg
                # elif self.heading_deg > 90:
                #     self.heading_deg = 450 - self.heading_deg
                # self.heading_deg +=30
                # if self.heading_deg >0 and self.heading_deg <
                # self.heading_deg = self.heading_deg % 360
                print(f"[GPS夹角] 相对于正东方向的夹角111: {self.heading_deg:.2f} 度")

                msg = {
                    "id": "imu",
                    "accelerometer": {"x": ax, "y": ay, "z": az},
                    "gyroscope": {"x": gx, "y": gy, "z": gz},
                    "heading_deg": round(self.heading_deg, 2)
                }
                sock_gnss_imu.sendto(json.dumps(msg).encode('utf-8'), (UDP_IP, UDP_PORT_GNSS_IMU))
                self.prev_timestamp = timestamp

        if 'LIDAR' in input_data and self.tcp_lidar_socket:
            lidar_data = input_data['LIDAR']
            xyz = lidar_data[1][:, :3]
            # 新增：对 LiDAR 点云进行镜像处理
            xyz = mirror_point_cloud(xyz)
            with self.lidar_buffer_lock:
                self.lidar_buffer.append(xyz)
                total_points = sum([arr.shape[0] for arr in self.lidar_buffer])

            if total_points >= self.lidar_batch_size:
                self.send_lidar_buffer()
                # A =0

        control = VehicleControl()
        with self.control_lock:
            if self.biaozhi:
                control.steer = self.control_command["steer"]
                control.throttle = self.control_command["throttle"]
                control.brake = self.control_command["brake"]
                print(f"[控制应用] Steer: {control.steer:.2f}, Throttle: {control.throttle:.2f}, Brake: {control.brake:.2f}")
            else:
                control.steer = 0
                control.throttle = 0
                control.brake = 100
                print(f"[控制应用] Steer: {control.steer:.2f}, Throttle: {control.throttle:.2f}, Brake: {control.brake:.2f}")
        return control

    def destroy(self):
        sock_gnss_imu.close()
        self.gps_file.close()
        if self.tcp_lidar_socket:
            self.send_lidar_buffer()  # 清理剩余数据
            self.tcp_lidar_socket.close()


def get_entry_point():
    return 'MyAgent'

