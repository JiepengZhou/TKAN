# 第一步，把pcap文件中的所有包提取出时序信息后存储进csv文件
import csv
import os
from datetime import datetime
from scapy.all import PcapReader, IP, TCP, UDP
from tqdm import tqdm
from collections import deque

file_name = 'Friday-WorkingHours.pcap'
folder_prefix = 'Fri'
output_folder = 'packetsdata_' + folder_prefix
print(f"current file : {file_name}")
print(f"output_folder : {output_folder}")

os.makedirs(output_folder, exist_ok=True)
# 批次大小
BATCH_SIZE = 100000
# CSV文件的基础名称
base_filename = 'packets_' + folder_prefix
# 字段顺序
FIELDS = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'timestamp', 'length', 'seq_num', 'ack_num', 'flags']

# 初始化批次计数器
batch_number = 0
# 用于暂存当前批次的包记录
current_batch = deque(maxlen=None)  # 使用deque提高添加和删除的效率
# 计数器，记录处理的总包数
total_packets = 0

# 打开pcap文件
with PcapReader(file_name) as packet_reader:
    for packet in tqdm(packet_reader, desc="Processing packets", unit="packet"):
        if IP in packet:
            ip_layer = packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            protocol = ip_layer.proto
            src_port = None
            dst_port = None
            seq_num = 0
            ack_num = 0
            flags = 0

            if TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                tcp_layer = packet[TCP]
                seq_num = tcp_layer.seq
                ack_num = tcp_layer.ack
                flags = tcp_layer.flags
            elif UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
                    
            # 打包时间戳,直接转成字符串再转成浮点数
            timestamp = float(str(packet.time))
            # 包的长度
            length = len(packet)
                
            # 创建记录
            record = {
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'timestamp': timestamp,
                'length': length,
                'seq_num': seq_num,
                'ack_num': ack_num,
                'flags': flags
            }
            current_batch.append(record)
                
            # 处理完成后增加计数器
            total_packets += 1
            # 当达到批次大小时，写入文件
            if total_packets % BATCH_SIZE == 0:
                batch_filename = os.path.join(output_folder, f"{base_filename}_{batch_number:04d}.csv")
                with open(batch_filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(current_batch)
                # 清空当前批次
                current_batch.clear()
                batch_number += 1

    # 处理剩余的包
    if current_batch:
        batch_filename = os.path.join(output_folder, f"{base_filename}_{batch_number:04d}.csv")
        with open(batch_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(current_batch)