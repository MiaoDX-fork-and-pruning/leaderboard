import socket
import struct
import time
import math  # 添加math模块的导入

# 定义数据类型常量
DataType_SteeringCmd = 1
DataType_TrqBreCmd = 2

class UDPSender:
    def __init__(self, target_ip="127.0.0.1", target_port=5005):
        self.target_ip = target_ip
        self.target_port = target_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_steering_cmd(self, steering_angle):
        header = struct.pack('>BB', DataType_SteeringCmd, 4)
        data_body = struct.pack('>f', steering_angle)
        packet = header + data_body
        self.socket.sendto(packet, (self.target_ip, self.target_port))

    def send_trq_bre_cmd(self, bre_enable, bre_value, trq_enable, trq_value):
        header = struct.pack('>BB', DataType_TrqBreCmd, 10)
        data_body = struct.pack('>BfBf', bre_enable, bre_value, trq_enable, trq_value)
        packet = header + data_body
        self.socket.sendto(packet, (self.target_ip, self.target_port))

    def close(self):
        self.socket.close()

if __name__ == "__main__":
    sender = UDPSender()
    send_interval = 0.1  # 发送间隔（秒），可根据需求调整（建议≥0.01秒）
    counter = 0

    try:
        print("开始循环发送控制命令（按Ctrl+C终止）...")
        while True:
            # 示例：发送动态变化的转向角度（从-1到1循环）
            steering_angle = math.sin(time.time())  # 正弦波变化（-1~1）
            bre_value = 0.0 if steering_angle > 0 else 0.1  # 简单逻辑：转向时轻微刹车
            trq_value = 0.5  # 固定油门值

            sender.send_steering_cmd(steering_angle)
            sender.send_trq_bre_cmd(
                bre_enable=1 if bre_value > 0 else 0,
                bre_value=bre_value,
                trq_enable=1,
                trq_value=trq_value
            )

            counter += 1
            print(f"[发送第{counter}次] 转向：{steering_angle:.2f} | 油门：{trq_value:.2f} | 刹车：{bre_value:.2f}")
            time.sleep(send_interval)  # 控制发送频率

    except KeyboardInterrupt:
        print("\n接收到终止信号，关闭连接...")
    finally:
        sender.close()
