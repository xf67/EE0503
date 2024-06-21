import argparse
import psutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def get_process_memory_usage(pid):
    try:
        proc = psutil.Process(pid)
        memory_info = proc.memory_info()  # 获取内存信息
        memory_usage = memory_info.rss / (1024 * 1024)  # 使用 RSS（Resident Set Size）
        print(f"Memory Usage at {time.time()}: {memory_usage} MB")  # 调试输出
        return memory_usage
    except psutil.NoSuchProcess:
        print(f"No such process with PID {pid}")
        return None
    except Exception as e:
        print(f"Error retrieving memory usage: {e}")
        return None

def main(pid):
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    start_time = time.time()
    ln, = plt.plot([], [], 'b-', animated=True)

    # 获取初始内存使用值
    initial_mem_usage = get_process_memory_usage(pid)
    if initial_mem_usage is None:
        print("Unable to retrieve initial memory usage.")
        return

    def init():
        ax.set_xlim(0, 300)
        ax.set_ylim(initial_mem_usage - 50, initial_mem_usage + 50)  # 根据初始值设置Y轴范围
        return ln,

    def update(frame):
        mem_usage = get_process_memory_usage(pid)
        if mem_usage is not None:
            current_time = time.time() - start_time
            xdata.append(current_time)
            ydata.append(mem_usage)
            if current_time > ax.get_xlim()[1]:
                ax.set_xlim(0, current_time)  # 动态调整X轴范围
            if mem_usage > ax.get_ylim()[1] or mem_usage < ax.get_ylim()[0]:
                ax.set_ylim(min(ydata) - 10, max(ydata) + 10)  # 动态调整Y轴范围
            ln.set_data(xdata, ydata)
        return ln,

    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=200)

    plt.xlabel('Time (s)')
    plt.ylabel('Memory Usage (MB)')
    plt.title(f'Memory Usage of Process {pid}')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Monitor memory usage of a process by PID.")
    parser.add_argument('pid', type=int, help="Process ID (PID) to monitor.")
    args = parser.parse_args()

    main(args.pid)