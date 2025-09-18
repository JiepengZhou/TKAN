# 第四步，将分割出来的flow文件和packet文件进行匹配，此时包比较少，可以节省很多时间
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm

def match_packets_to_flows(flow_df, packet_df):
    """将包匹配到流"""
    # 初始化滑动窗口指针
    flow_index = 0
    
    # 初始化活动流哈希表
    active_flows = {}
    
    # 滑动窗口遍历Packet文件
    matched_packets = []
    # 获取包的总数
    total_packets = len(packet_df)
    with tqdm(total=total_packets, desc="Matching packets to flows") as pbar:
        for _, packet in packet_df.iterrows():
            current_time = packet['timestamp']
        
            # 加载当前时间之后的流到哈希表
            while flow_index < len(flow_df) and flow_df['start_timestamp'].iloc[flow_index] <= current_time:
                row = flow_df.iloc[flow_index]
                key = (
                    row['Source IP'],
                    row['Destination IP'],
                    row['Source Port'],
                    row['Destination Port']
                )
                # 处理同一四元组的不同流
                if key in active_flows:
                    old_flow = active_flows[key]
                    if old_flow['end_timestamp'] < current_time:
                        del active_flows[key]
                active_flows[key] = {
                    'flow_id': row['flow_id'],
                    'label': row['Label'],
                    'end_timestamp': row['end_timestamp']
                }
                flow_index += 1
        
            # 清理已过期的流
            expired_keys = [key for key in active_flows if active_flows[key]['end_timestamp'] < current_time]
            for key in expired_keys:
                del active_flows[key]
        
            # 尝试匹配流
            flow_key = (
                packet['src_ip'],
                packet['dst_ip'],
                packet['src_port'],
                packet['dst_port']
            )
        
            if flow_key in active_flows:
                info = active_flows[flow_key]
                matched_packet = {
                    **packet.to_dict(),
                    'flow_id': info['flow_id'],
                    'label': info['label']
                }
                matched_packets.append(matched_packet)
            else:
                # 检查反向键
                reversed_key = (
                    packet['dst_ip'],
                    packet['src_ip'],
                    packet['dst_port'],
                    packet['src_port']
                )
                if reversed_key in active_flows:
                    info = active_flows[reversed_key]
                    matched_packet = {
                        **packet.to_dict(),
                        'flow_id': info['flow_id'],
                        'label': info['label']
                    }
                    matched_packets.append(matched_packet)
            # 更新进度条
            pbar.update(1)
    print(f"matched_packet Len: {len(matched_packets)}")
    return matched_packets

def save_matched_packets(matched_packets, output_folder, batch_size=10000):
    """保存匹配结果到 CSV 文件"""
    print("Saving matched packets...")
    # 转换为 DataFrame
    matched_packet_df = pd.DataFrame(matched_packets)
    print(matched_packet_df.head(10))
    # 按照flow_id和时间戳排序
    matched_packet_df.sort_values(by=['flow_id', 'timestamp'], inplace=True)
    
    # 分组保存匹配结果
    grouped = matched_packet_df.groupby('flow_id')
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历所有flow_id，并合并每10000个flow_id到一个DataFrame
    current_flows = []    # 保存当前批次的flow_id分组
    flow_count = 0        # 记录当前批次的flow_id数量
    flow_number = 0       # 记录总的flow_id数量
    
    for flow_id, group in tqdm(grouped, desc="Saving flows"):
        current_flows.append(group)
        flow_count += 1
        
        if flow_count == batch_size:
            # 合并当前批次的flow_id
            merged_df = pd.concat(current_flows, ignore_index=True)
            
            # 生成文件名
            start_number = flow_number * batch_size
            end_number = flow_number * batch_size + batch_size - 1
            filename = os.path.join(output_folder, f'flows_{start_number}-{end_number}.csv')
            
            # 保存到CSV文件
            merged_df.to_csv(filename, index=False)
            
            # 重置批次变量
            current_flows = []
            flow_count = 0
            flow_number += 1  # 流数增加
    
    # 处理剩余的flow_id
    if flow_count > 0:
        merged_df = pd.concat(current_flows, ignore_index=True)
        start_number = flow_number * batch_size
        end_number = flow_number * batch_size + flow_count - 1
        filename = os.path.join(output_folder, f'flows_{start_number}-{end_number}.csv')
        merged_df.to_csv(filename, index=False)
    print("Matched packets saved successfully!")

def main():
    # 读取对应的文件
    flow_df = pd.read_csv('Dataset/Fri_Aft_filtered_flow_file.csv')
    
    # 读取Packet文件
    # packet_df = pd.read_csv('Dataset/Fri_Aft_DDoS_filtered_packet_file.csv')
    packets_folder = "Dataset/DDoSDataset"
    packet_files = [
        os.path.join(packets_folder, f)
        for f in os.listdir(packets_folder)
        if f.startswith('filtered_packets_') and f.endswith('.csv')
    ]
    packet_df = pd.DataFrame()
    for file in tqdm(packet_files, desc="Loading packet files"):
        temp_df = pd.read_csv(file)
        packet_df = pd.concat([packet_df, temp_df], ignore_index=True)

    print(f"flow_df Timestamp Max: {flow_df['Timestamp'].max()}, Timestamp Min: {flow_df['Timestamp'].min()}")
    print(f"packet_df Timestamp Max: {packet_df['timestamp'].max()}, Timestamp Min: {packet_df['timestamp'].min()}")
    
    # 匹配包到流
    matched_packets = match_packets_to_flows(flow_df, packet_df)
    
    # 保存匹配结果
    output_folder = "Dataset/DDoS"
    os.makedirs(output_folder, exist_ok=True)
    save_matched_packets(matched_packets, 'Dataset/DDoS')
    
    print("Process completed!")

if __name__ == "__main__":
    main()