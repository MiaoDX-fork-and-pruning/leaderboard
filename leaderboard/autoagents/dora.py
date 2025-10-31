import socket
import json
import math
import numpy as np
import threading
import struct
import os
import cv2
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from carla import VehicleControl
try:
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
except ImportError:
    CarlaDataProvider = None

# 定义网络相关参数
# For Docker: send to dora-agent container, bind to all interfaces for receiving
TCP_IP = "dora-agent"  # Send to dora-agent container
UDP_IP = "dora-agent"  # Send to dora-agent container
UDP_CONTROL_IP = "0.0.0.0"  # Bind to all interfaces to receive control
UDP_PORT_GNSS_IMU = 12345
UDP_PORT_CONTROL = 23456
TCP_PORT_LIDAR = 5005
TCP_PORT_ROUTE_METADATA = 5006

# 创建用于接收 GNSS 和 IMU 数据的 UDP 套接字
sock_gnss_imu = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

EARTH_RADIUS_METERS = 6378137.0

# 将弧度转换为角度
def deg_heading_from_compass(compass_radian):
    return math.degrees(compass_radian)

def _geodesic_displacement_meters(prev_lat, prev_lon, lat, lon):
    """Compute planar displacement (dx, dy) in meters between two geodetic points."""
    lat1 = math.radians(prev_lat)
    lat2 = math.radians(lat)
    d_lat = lat2 - lat1
    d_lon = math.radians(lon - prev_lon)
    avg_lat = 0.5 * (lat1 + lat2)
    dx = d_lon * math.cos(avg_lat) * EARTH_RADIUS_METERS
    dy = d_lat * EARTH_RADIUS_METERS
    return dx, dy


def _serialize_vector(vector):
    return {"x": vector.x, "y": vector.y, "z": vector.z}


def _serialize_transform(transform):
    return {
        "location": _serialize_vector(transform.location),
        "rotation": {
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw,
            "roll": transform.rotation.roll
        }
    }


def _serialize_bounding_box(box):
    return {
        "location": _serialize_vector(box.location),
        "extent": _serialize_vector(box.extent),
        "rotation": {
            "pitch": box.rotation.pitch,
            "yaw": box.rotation.yaw,
            "roll": box.rotation.roll
        }
    }

class MyAgent(AutonomousAgent):
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        self.full_global_plan_gps = global_plan_gps
        self.full_global_plan_world = global_plan_world_coord
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._prepare_route_metadata()

    def setup(self, path_to_conf_file):
        # 设置传感器赛道
        self.track = Track.SENSORS
        # 初始化当前航向角
        self.current_heading = 0.0
        # 初始化上一时刻时间戳
        self.prev_timestamp = None
        # 初始化相对于正东方向的夹角
        self.heading_deg = 0.0
        # 初始化控制指令
        self.control_command = {"steer": 0.0, "throttle": 0.0, "brake": 1.0}
        # 初始化控制指令锁
        self.control_lock = threading.Lock()
        # 打开 GPS 路径文件
        self.gps_file = open("gps_path.txt", "w")
        # LiDAR 发送锁
        self.lidar_lock = threading.Lock()
        # 运动学状态缓存
        self.prev_state_timestamp = None
        self.prev_gnss = None
        self.prev_velocity_vector = None
        self.last_frame_id = 0
        # 路径与地图数据缓存
        if not hasattr(self, "map_route_payload"):
            self.map_route_payload = None
        if not hasattr(self, "map_route_sent"):
            self.map_route_sent = False
        # 启动接收控制指令的线程
        threading.Thread(target=self.receive_control_loop, daemon=True).start()
        # 初始化 TCP LiDAR 套接字
        self.tcp_lidar_socket = None
        # 启动连接 TCP LiDAR 的线程
        threading.Thread(target=self.connect_tcp_lidar, daemon=True).start()
        # 启动地图与路径发布线程
        threading.Thread(target=self.publish_route_metadata, daemon=True).start()

    def sensors(self):
        # 定义传感器配置
        sensors = [
            {"type": "sensor.other.gnss", "id": "GPS", "x": 0, "y": 0, "z": 1.60, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            {"type": "sensor.other.imu", "id": "IMU", "x": 0, "y": 0, "z": 1.60, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            {"type": "sensor.lidar.ray_cast", "id": "LIDAR", "x": 0, "y": 0.0, "z": 2.5, "roll": 0, "pitch": 0, "yaw": -90,
             "range": 80, "rotation_frequency": 10, "channels": 64, "upper_fov": 10, "lower_fov": -25, "points_per_second": 600000},
            {"type": "sensor.speedometer", "id": "SPEED"}
        ]

        # Add RGB camera for visualization when VIS_DORA=1
        if os.getenv('VIS_DORA') == '1':
            sensors.append({
                "type": "sensor.camera.rgb",
                "id": "RGB",
                "x": -1.5,
                "y": 0.0,
                "z": 2.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 1024,
                "height": 256,
                "fov": 110
            })

        return sensors

    def connect_tcp_lidar(self):
        # 尝试连接 TCP LiDAR 服务器
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

    def send_lidar_frame(self, xyz: np.ndarray) -> None:
        if self.tcp_lidar_socket is None:
            return

        with self.lidar_lock:
            scaled_xyz = (xyz * 100).astype(np.int16, copy=False)
            data_bytes = scaled_xyz.tobytes()

        header = struct.pack("!I", len(data_bytes))
        try:
            self.tcp_lidar_socket.sendall(header + data_bytes)
            print(f"[LIDAR] 已发送 {scaled_xyz.shape[0]} 个点（带帧头）")
        except Exception as e:
            print(f"[LIDAR 发送错误] {e}")
            self.tcp_lidar_socket = None
            threading.Thread(target=self.connect_tcp_lidar, daemon=True).start()

    def publish_route_metadata(self):
        import time
        while not self.map_route_sent:
            if not self.map_route_payload:
                time.sleep(0.5)
                continue
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((TCP_IP, TCP_PORT_ROUTE_METADATA))
                payload_bytes = json.dumps(self.map_route_payload).encode('utf-8')
                header = struct.pack("!I", len(payload_bytes))
                sock.sendall(header + payload_bytes)
                sock.close()
                self.map_route_sent = True
                print(f"[路线元数据] 已发送，大小 {len(payload_bytes)} 字节")
            except Exception as e:
                print(f"[路线元数据] 发送失败，重试中... ({e})")
                time.sleep(1)

    def _prepare_route_metadata(self):
        route_world = []
        route_gps = []

        if hasattr(self, "full_global_plan_world"):
            for transform, road_option in self.full_global_plan_world:
                route_world.append({
                    "location": {
                        "x": transform.location.x,
                        "y": transform.location.y,
                        "z": transform.location.z
                    },
                    "rotation": {
                        "pitch": transform.rotation.pitch,
                        "yaw": transform.rotation.yaw,
                        "roll": transform.rotation.roll
                    },
                    "road_option": str(road_option)
                })

        if hasattr(self, "full_global_plan_gps"):
            for gps_point, road_option in self.full_global_plan_gps:
                route_gps.append({
                    "lat": gps_point.get("lat"),
                    "lon": gps_point.get("lon"),
                    "alt": gps_point.get("z"),
                    "road_option": str(road_option)
                })

        opendrive_text = None
        if CarlaDataProvider is not None:
            try:
                opendrive_text = CarlaDataProvider.get_map().to_opendrive()
            except Exception as exc:
                print(f"[路线元数据] 获取 OpenDRIVE 失败: {exc}")

        self.map_route_payload = {
            "id": "route_metadata",
            "map_opendrive": opendrive_text,
            "global_plan_world": route_world,
            "global_plan_gps": route_gps
        }
        self.map_route_sent = False

    def _serialize_vehicle(self, vehicle, hero_id):
        try:
            control = vehicle.get_control()
        except Exception:
            control = None
        attributes = {}
        try:
            for key, value in vehicle.attributes.items():
                attributes[key] = getattr(value, "value", value)
        except Exception:
            attributes = {}
        return {
            "id": vehicle.id,
            "type_id": vehicle.type_id,
            "is_hero": vehicle.id == hero_id,
            "transform": _serialize_transform(vehicle.get_transform()),
            "velocity": _serialize_vector(vehicle.get_velocity()),
            "angular_velocity": _serialize_vector(vehicle.get_angular_velocity()),
            "acceleration": _serialize_vector(vehicle.get_acceleration()),
            "bounding_box": _serialize_bounding_box(vehicle.bounding_box),
            "attributes": attributes,
            "control": {
                "steer": control.steer if control else 0.0,
                "throttle": control.throttle if control else 0.0,
                "brake": control.brake if control else 0.0,
                "hand_brake": bool(control.hand_brake) if control else False,
                "reverse": bool(control.reverse) if control else False,
                "gear": int(control.gear) if control else 0
            } if control else None
        }

    def _serialize_walker(self, walker):
        try:
            control = walker.get_control()
        except Exception:
            control = None

        return {
            "id": walker.id,
            "type_id": walker.type_id,
            "transform": _serialize_transform(walker.get_transform()),
            "velocity": _serialize_vector(walker.get_velocity()),
            "angular_velocity": _serialize_vector(walker.get_angular_velocity()),
            "bounding_box": _serialize_bounding_box(walker.bounding_box),
            "control": {
                "direction": {
                    "x": control.direction.x,
                    "y": control.direction.y,
                    "z": control.direction.z
                },
                "speed": control.speed
            } if control else None
        }

    def _serialize_traffic_light(self, traffic_light):
        trigger = traffic_light.trigger_volume
        state = getattr(traffic_light.state, "name", str(traffic_light.state))
        return {
            "id": traffic_light.id,
            "type_id": traffic_light.type_id,
            "transform": _serialize_transform(traffic_light.get_transform()),
            "trigger_volume": {
                "location": _serialize_vector(trigger.location),
                "rotation": {
                    "pitch": trigger.rotation.pitch,
                    "yaw": trigger.rotation.yaw,
                    "roll": trigger.rotation.roll
                },
                "extent": _serialize_vector(trigger.extent)
            },
            "state": state,
            "is_frozen": bool(traffic_light.is_frozen)
        }

    def _serialize_stop_sign(self, stop_sign):
        trigger = stop_sign.trigger_volume
        return {
            "id": stop_sign.id,
            "type_id": stop_sign.type_id,
            "transform": _serialize_transform(stop_sign.get_transform()),
            "trigger_volume": {
                "location": _serialize_vector(trigger.location),
                "rotation": {
                    "pitch": trigger.rotation.pitch,
                    "yaw": trigger.rotation.yaw,
                    "roll": trigger.rotation.roll
                },
                "extent": _serialize_vector(trigger.extent)
            }
        }

    def _collect_actor_snapshot(self, timestamp):
        if CarlaDataProvider is None:
            return None
        try:
            world = CarlaDataProvider.get_world()
            hero = CarlaDataProvider.get_hero_actor()
        except Exception as exc:
            print(f"[演员快照] 访问世界失败: {exc}")
            return None

        if world is None:
            return None

        actor_list = world.get_actors()
        hero_id = hero.id if hero else None

        vehicles = [self._serialize_vehicle(actor, hero_id) for actor in actor_list.filter('*vehicle*')]
        walkers = [self._serialize_walker(actor) for actor in actor_list.filter('*walker*')]
        traffic_lights = [self._serialize_traffic_light(actor) for actor in actor_list.filter('*traffic_light*')]
        stop_signs = [self._serialize_stop_sign(actor) for actor in actor_list.filter('*traffic.stop*')]

        return {
            "id": "actors",
            "timestamp": timestamp,
            "frame": getattr(self, "last_frame_id", 0),
            "vehicles": vehicles,
            "walkers": walkers,
            "traffic_lights": traffic_lights,
            "stop_signs": stop_signs
        }

    def receive_control_loop(self):
        # 接收控制指令
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


        if 'GPS' in input_data:
            gps_data = input_data['GPS']
            x, y, z = gps_data[1]
            msg = {"id": "gnss", "x": x, "y": y, "z": z}
            print(f"[GNSS] x: {x:.12f}, y: {y:.12f}")
            sock_gnss_imu.sendto(json.dumps(msg).encode('utf-8'), (UDP_IP, UDP_PORT_GNSS_IMU))
            self.gps_file.write(f"{x} {y}\n")

        if 'IMU' in input_data:
            imu_data = input_data['IMU']
            imu_values = imu_data[1]
            if len(imu_values) == 7:
                ax, ay, az, gx, gy, gz, heading_rad = imu_values
                heading_deg = deg_heading_from_compass(heading_rad)
                print(f"[IMU] heading_deg (from rad): {heading_deg:.2f}")
                self.heading_deg = heading_deg
                msg = {
                    "id": "imu",
                    "accelerometer": {"x": ax, "y": ay, "z": az},
                    "gyroscope": {"x": gx, "y": gy, "z": gz},
                    "heading_rad": heading_rad,
                    "heading_deg": heading_deg
                }
                sock_gnss_imu.sendto(json.dumps(msg).encode('utf-8'), (UDP_IP, UDP_PORT_GNSS_IMU))
                self.prev_timestamp = timestamp

        if 'GPS' in input_data and 'SPEED' in input_data and self.heading_deg is not None:
            gps_frame, gps_values = input_data['GPS']
            speed_measurement = input_data['SPEED'][1]
            lat, lon, alt = gps_values
            speed = float(speed_measurement.get("speed", 0.0))
            velocity_vector = None
            acceleration_vector = None

            if self.prev_gnss is not None and self.prev_state_timestamp is not None:
                dt = max(timestamp - self.prev_state_timestamp, 1e-3)
                dx, dy = _geodesic_displacement_meters(self.prev_gnss[0], self.prev_gnss[1], lat, lon)
                dz = alt - self.prev_gnss[2]
                vx = dx / dt
                vy = dy / dt
                vz = dz / dt
                velocity_vector = {"x": vx, "y": vy, "z": vz}

                if self.prev_velocity_vector is not None:
                    ax = (vx - self.prev_velocity_vector[0]) / dt
                    ay = (vy - self.prev_velocity_vector[1]) / dt
                    az = (vz - self.prev_velocity_vector[2]) / dt
                    acceleration_vector = {"x": ax, "y": ay, "z": az}

                self.prev_velocity_vector = (vx, vy, vz)

            state_msg = {
                "id": "state",
                "frame": int(gps_frame),
                "timestamp": timestamp,
                "gnss": {"lat": lat, "lon": lon, "alt": alt},
                "heading_deg": self.heading_deg,
                "speed": speed
            }

            if velocity_vector:
                state_msg["velocity"] = velocity_vector
            if acceleration_vector:
                state_msg["acceleration"] = acceleration_vector

            sock_gnss_imu.sendto(json.dumps(state_msg).encode('utf-8'), (UDP_IP, UDP_PORT_GNSS_IMU))

            self.prev_gnss = (lat, lon, alt)
            self.prev_state_timestamp = timestamp
            self.last_frame_id = int(gps_frame)

        if 'LIDAR' in input_data:
            lidar_data = input_data['LIDAR']
            xyz = lidar_data[1][:, :3]
            self.send_lidar_frame(xyz)

        # Save RGB camera images for visualization when VIS_DORA=1
        if 'RGB' in input_data:
            rgb_frame_id, rgb_image = input_data['RGB']

            # Save only every 5th frame to reduce disk I/O and storage
            if rgb_frame_id % 5 == 0:
                # CARLA camera returns BGRA, convert to RGB
                rgb_image = rgb_image[:, :, :3]

                # Save to logs/vis_dora directory
                vis_dir = '/workspace/logs/vis_dora'
                os.makedirs(vis_dir, exist_ok=True)
                save_path = os.path.join(vis_dir, f'rgb_{rgb_frame_id:06d}.png')

                # Convert RGB to BGR for cv2.imwrite
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, bgr_image)
                print(f"[VIS_DORA] Saved RGB image: {save_path}")

        actor_snapshot = self._collect_actor_snapshot(timestamp)
        if actor_snapshot:
            try:
                sock_gnss_imu.sendto(json.dumps(actor_snapshot).encode('utf-8'), (UDP_IP, UDP_PORT_GNSS_IMU))
            except Exception as exc:
                print(f"[演员快照] 发送失败: {exc}")

        control = VehicleControl()
        with self.control_lock:
            control.steer = self.control_command["steer"]
            control.throttle = self.control_command["throttle"]
            control.brake = self.control_command["brake"]
            print(f"[控制应用] Steer: {control.steer:.2f}, Throttle: {control.throttle:.2f}, Brake: {control.brake:.2f}")
        return control

    def destroy(self):
        # 清理资源
        sock_gnss_imu.close()
        self.gps_file.close()
        if self.tcp_lidar_socket:
            self.tcp_lidar_socket.close()

def get_entry_point():
    return 'MyAgent'
    
