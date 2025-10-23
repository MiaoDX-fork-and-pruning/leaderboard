import socket
import json
import numpy as np
import open3d as o3d

# 定义接收GNSS和IMU数据的端口
UDP_PORT_GNSS_IMU = 5006
# 定义接收LiDAR数据的端口
UDP_PORT_LIDAR = 5007
# 定义本地IP地址
UDP_IP = "127.0.0.1"

# 创建用于接收GNSS和IMU数据的套接字
sock_gnss_imu = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_gnss_imu.bind((UDP_IP, UDP_PORT_GNSS_IMU))

# 创建用于接收LiDAR数据的套接字
sock_lidar = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_lidar.bind((UDP_IP, UDP_PORT_LIDAR))

# 用于存储接收到的LiDAR点云数据
point_cloud_data = []

while True:
    # 接收GNSS和IMU数据
    data_gnss_imu, addr_gnss_imu = sock_gnss_imu.recvfrom(1024)  # 假设数据大小不超过1024字节
    try:
        json_data = json.loads(data_gnss_imu.decode('utf-8'))
        if json_data["id"] == "gnss":
            print(f"[Received GNSS] Latitude: {json_data['latitude']}, Longitude: {json_data['longitude']}, Altitude: {json_data['altitude']}")
        elif json_data["id"] == "imu":
            print(f"[Received IMU] Accelerometer: ({json_data['accelerometer']['x']}, {json_data['accelerometer']['y']}, {json_data['accelerometer']['z']})")
            print(f"[Received IMU] Gyroscope: ({json_data['gyroscope']['x']}, {json_data['gyroscope']['y']}, {json_data['gyroscope']['z']})")
            print(f"[Received IMU] Heading Degrees: {json_data['heading_deg']}")
    except json.JSONDecodeError:
        print("[警告] 无法解析接收到的GNSS或IMU数据")

    # 接收LiDAR数据
    data_lidar, addr_lidar = sock_lidar.recvfrom(65536)  # 接收LiDAR数据，缓冲区大小可根据实际情况调整
    if not data_lidar:
        break
    # 将接收到的二进制数据转换为numpy数组
    chunk = np.frombuffer(data_lidar, dtype=np.float32)
    # 假设每个点是3维坐标，将数组重塑为点的形式
    points = chunk.reshape(-1, 3)
    point_cloud_data.append(points)

    # 每接收到一定数量的数据后保存为PCD文件
    if len(point_cloud_data) >= 10:  # 这里可以根据需要调整数量
        all_points = np.vstack(point_cloud_data)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        # 生成文件名，可以根据时间来命名，确保文件名唯一
        from time import time
        timestamp = str(int(time()))
        file_name = f"received_lidar_{timestamp}.pcd"
        o3d.io.write_point_cloud(file_name, pcd)
        print(f"已保存点云数据为 {file_name}")
        point_cloud_data = []  # 清空已保存的点云数据列表

# 关闭套接字
sock_gnss_imu.close()
sock_lidar.close()
