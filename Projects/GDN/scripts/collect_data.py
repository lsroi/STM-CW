import os
import csv
import time
from datetime import datetime, timedelta
from urllib import parse
import requests
import random


# 定义服务列表
SERVICES = [
    "frontend", "cartservice", "productcatalogservice",
    "currencyservice", "paymentservice", "shippingservice",
    "emailservice", "checkoutservice", "recommendationservice",
    "adservice", "loadgenerator"
]

# 服务对应的 POD
PODS = [
    "frontend-754cdbf884-gbzg5",
    "cartservice-f84bf7dd4-5dqfc",
    "productcatalogservice-59cf6fd7b5-mdnw7",
    "currencyservice-84459c6759-hnvjd",
    "paymentservice-5575668b5c-qkkhs",
    "shippingservice-fb4c9695c-b9hht",
    "emailservice-6fb4dd89fc-bf6rl",
    "checkoutservice-5d9894c787-7dr2g",
    "recommendationservice-589895488f-bgbz8",
    "adservice-8568877bf9-nt5sn",
    "loadgenerator-5d8745bc7-tdx74"
]

# 定义要收集的核心指标（根据您的需求调整）
CORE_METRICS = [
    "cpu_usage", "cpu_system_usage", "cpu_user_usage",
    "memory_usage", "memory_max_usage",
    "pod_restarts_total",
]
# 核心指标2，不以POD为单位，是整体的指标
CORE_METRICS2 = [
    "fs_reads_total", "fs_writes_total",
    "fs_read_seconds", "fs_write_seconds",
    "fs_usage_bytes", "fs_limit_bytes",
    "network_receive_bridge", "network_receive_eth0",
    "network_transmit_bridge", "network_transmit_eth0",
    "network_receive_errors_bridge", "network_receive_errors_eth0",
    "network_transmit_errors_bridge", "network_transmit_errors_eth0",
]

#    "fs_reads", "fs_writes", "fs_read_seconds", "fs_write_seconds",
#    "disk_read", "disk_write",
#    "rps", "p50_rt", "p99_rt", "max_rt", "error_rate"

# 构建宽表表头（timestamp + 每个服务的所有指标）
def create_wide_table_header():
    header = ["timestamp"]
    for service in SERVICES:
        for metric in CORE_METRICS:
            header.append(f"{service}_{metric}")
    for metric in CORE_METRICS2:
        header.append(metric)
    return header

# 生成时间戳
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def time_format(tt):
    timeArray = time.strptime(tt, "%Y-%m-%d %H:%M:%S")   #转换成时间数组
    timestamp = time.mktime(timeArray)                   #转换成时间戳
    return int(timestamp)

def clear_csv(file):
    # 如果文件不存在，创建一个空文件
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass  # 创建空文件
    else:
        # 文件存在时，用 r+ 打开并清空
        with open(file, "r+") as f:
            f.truncate(0)


def get_data(sql, start_time, end_time):
    url = f'http://127.0.0.1:60925//api/v1/query_range?query={sql}&start={time_format(start_time)}&end={time_format(end_time)}&step=30&timeout=40s'
    res = requests.get(url=url)

    # 检查响应是否成功
    if res.status_code != 200:
        print(f"API请求失败，状态码: {res.status_code}")
        # 记录完整的请求URL（用于调试）
        print(f"请求URL: {url}")
        return 0  # 或者根据需要返回其他默认值或抛出异常

    # 解析JSON数据
    data = res.json()

    # 检查结果结构是否符合预期
    if "data" not in data or "result" not in data["data"]:
        print("API响应格式异常:", data)
        return 0

    # 检查result列表是否为空
    if not data["data"]["result"]:
        print(f"查询无结果: {sql}")
        return 0

    vmax = 0
    try:
        for value in data["data"]["result"][0]["values"]:
            vd = float(value[1])
            if vd >= vmax:
                vmax = vd
    except (KeyError, ValueError) as e:
        print(f"处理查询结果出错: {e}")
        return 0
    return round(vmax,6)

# 模拟从Prometheus获取服务指标
def fetch_service_metrics(service_name, pod, start_time, end_time):
    metrics = {}
    for metric in CORE_METRICS:
        flag = True
        # 模拟指标值（实际应调用Prometheus API）
        if "cpu_usage" == metric:
            # CPU 使用率（百分比）
            sql = f'rate(container_cpu_usage_seconds_total{{pod="{pod}"}}[1m]) * 100'
        elif "cpu_system_usage" == metric:
            # 内核态 CPU 使用率，反映系统调用开销
            sql = f'rate(container_cpu_system_seconds_total{{pod="{pod}"}}[1m]) * 100'
        elif "cpu_user_usage" == metric:
            # 用户态 CPU 使用率，反映应用自身计算负载
            sql = f'rate(container_cpu_user_seconds_total{{pod="{pod}"}}[1m]) * 100'
        elif "memory_usage" == metric:
            # 内存使用率（百分比，总内存7.62GiB）
            sql  = f'container_memory_usage_bytes{{pod="{pod}"}} / (1048576 * 1024 * 7.62) * 100'
        elif "memory_max_usage" == metric:
            # 内存最大使用率
            sql = f'container_memory_max_usage_bytes{{pod="{pod}"}} / (1048576 * 1024 * 7.62) * 100'
        elif "pod_restarts_total" == metric:
            # 容器重启次数，高重启次数可能提示应用不稳定或资源不足
            sql = f'kube_pod_container_status_restarts_total{{pod="{pod}"}}'
        else:
            flag = False
        if flag is True:
            metrics[metric] = get_data(sql, start_time, end_time)

    return metrics

# 获取非POD的核心指标
def fetch_metrics(start_time, end_time):
    metrics = {}
    for metric in CORE_METRICS2:
        flag = True
        # 模拟指标值（实际应调用Prometheus API）
        if "fs_reads_total" == metric:
            # 读次数
            sql = f'container_fs_reads_total{{device="/dev/sdd"}}'
        elif "fs_writes_total" == metric:
            # 写次数
            sql = f'container_fs_writes_total{{device="/dev/sdd"}}'
        elif "fs_read_seconds" == metric:
            # 读耗时
            sql = f'container_fs_read_seconds_total{{device="/dev/sdd"}}'
        elif "fs_write_seconds" == metric:
            # 写耗时
            sql = f'container_fs_write_seconds_total{{device="/dev/sdd"}}'
        elif "fs_usage_bytes" == metric:
            # 已用空间
            sql = f'container_fs_usage_bytes{{device="/dev/sdd"}}'
        elif "fs_limit_bytes" == metric:
            # 空间限制
            sql = f'container_fs_limit_bytes{{device="/dev/sdd"}}'
        elif "network_receive_bridge" == metric:
            # bridge POD间网络性能和通信流量(接收流量
            sql = f'container_network_receive_bytes_total{{interface="bridge"}}'
        elif "network_receive_eth0" == metric:
            # eth0 整体网络负载和物理接口状态(接收流量
            sql = f'container_network_receive_bytes_total{{interface="eth0"}}'
        elif "network_transmit_bridge" == metric:
            # 发送流量
            sql = f'container_network_transmit_bytes_total{{interface="bridge"}}'
        elif "network_transmit_eth0" == metric:
            # 发送流量
            sql = f'container_network_transmit_bytes_total{{interface="eth0"}}'
        elif "network_receive_errors_bridge" == metric:
            # 接收错误数
            sql = f'container_network_receive_errors_total{{interface="bridge"}}'
        elif "network_receive_errors_eth0" == metric:
            # 接收错误数
            sql = f'container_network_receive_errors_total{{interface="eth0"}}'
        elif "network_transmit_errors_bridge" == metric:
            # 发送错误数
            sql = f'container_network_transmit_errors_total{{interface="bridge"}}'
        elif "network_transmit_errors_eth0" == metric:
            # 发送错误数
            sql = f'container_network_transmit_errors_total{{interface="eth0"}}'
        else:
            flag = False
        if flag is True:
            metrics[metric] = get_data(sql, start_time, end_time)

    return metrics


# 收集所有服务的数据并写入CSV
def collect_and_write_data(start_time_str, end_time_str, interval_seconds, output_file="train_data.csv"):
    """
        按指定时间间隔收集数据
        参数:
        start_time_str: 开始时间字符串，格式: "YYYY-MM-DD HH:MM:SS"
        end_time_str: 结束时间字符串，格式: "YYYY-MM-DD HH:MM:SS"
        interval_minutes: 收集数据的时间间隔（分钟）
        output_file: 输出CSV文件路径
    """

    # 将字符串转换为datetime对象
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

    # 创建表头
    header = create_wide_table_header()

    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        # 当前时间点从start_time开始
        current_time = start_time

        # 循环收集每个时间点的数据
        while current_time <= end_time:
            # 将当前时间转换为字符串格式
            current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

            # 构建当前时间点的行数据
            row_data = [current_time_str]

            # 对于每个时间点，查询该时间点前后一小段时间的数据
            # 这里设置为前后各5秒钟
            query_start = (current_time - timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S")
            query_end = (current_time + timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S")

            # 收集每个服务的所有指标
            for service, pod in zip(SERVICES, PODS):
                metrics = fetch_service_metrics(service, pod, query_start, query_end)
                for metric in CORE_METRICS:
                    row_data.append(metrics.get(metric, 0))

            # 收集整体指标
            metrics = fetch_metrics(query_start, query_end)
            for metric in CORE_METRICS2:
                row_data.append(metrics.get(metric, 0))

            # 写入当前时间点的数据
            writer.writerow(row_data)

            # 移动到下一个时间点
            current_time += timedelta(seconds=interval_seconds)

            # 为避免API请求过于频繁，可以添加短暂延迟
            time.sleep(0.1)

    print(f"数据已写入 {output_file}")


if __name__ == '__main__':
    start_time = '2025-06-08 15:00:00'
    end_time = '2025-06-08 18:30:00'
    interval = 10  # 收集间隔（秒）

    clear_csv("train_data.csv")
    collect_and_write_data(start_time, end_time, interval)