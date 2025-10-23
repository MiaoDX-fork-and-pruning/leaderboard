import subprocess
import time

def run_script(script_name):
    """运行指定的脚本"""
    print(f"Starting {script_name}...")
    process = subprocess.Popen(['python', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def main():
    # 启动接收脚本
    receiver_process = run_script('rec_test.py')
    
    # 等待1秒，确保接收脚本已经启动
    time.sleep(1)
    
    # 启用发送脚本
    sender_process = run_script('send_test.py')
    
    try:
        # 等待发送脚本结束
        sender_process.wait()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # 终止接收脚本
        receiver_process.terminate()
        receiver_process.wait()
        print("Receiver stopped.")

if __name__ == '__main__':
    main()
