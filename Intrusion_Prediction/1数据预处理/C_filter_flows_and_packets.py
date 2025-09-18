# 第三步，将想要提取的时间范围先加8， 然后使用这个网站：https://tool.chinaz.com/tools/unixtime.aspx转化为UNIX时间戳，再将flow文件和包文件进行切割
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

def load_and_filter_flows(flows_folder, start_time, end_time, filtered_flow_file, max_flows=30000):
    """加载并过滤流数据，保存到 CSV 文件"""
    print("Loading and filtering flow data...")
    # 读取流文件
    flow_files = [
        os.path.join(flows_folder, f)
        for f in os.listdir(flows_folder)
        if f.startswith('Flow_') and f.endswith('.csv')
    ]
    
    flow_df = pd.DataFrame()
    for file in tqdm(flow_files, desc="Loading packet files"):
        temp_df = pd.read_csv(file)
        flow_df = pd.concat([flow_df, temp_df], ignore_index=True)

    # 打印最大最小时间戳：
    print(f"Flow_df_Timestamp_MAX (datetime): {flow_df['Timestamp'].max()}, Flow_df_Timestamp_MIN (datetime): {flow_df['Timestamp'].min()}")
    print(f"Flow_df_Initial_len: {len(flow_df)}")
    # 去除空格
    flow_df.columns = flow_df.columns.str.strip()
    # 添加索引
    flow_df['flow_id'] = flow_df.index
    
    # 处理流的时间信息
    print(f"FLow Duration MAX: {flow_df['Flow Duration'].max()}, MIN: {flow_df['Flow Duration'].min()}")
    flow_df['Flow Duration'] = flow_df['Flow Duration'] / 10**6
    print(f"FLow Duration MAX: {flow_df['Flow Duration'].max()}, MIN: {flow_df['Flow Duration'].min()}")
    
    flow_df['start_timestamp'] = flow_df['Timestamp']
    flow_df['end_timestamp'] = flow_df['start_timestamp'] + flow_df['Flow Duration']
    print(f"flow_df Start Timestamp Max: {flow_df['start_timestamp'].max()}, Min: {flow_df['start_timestamp'].min()}")
    print(f"flow_df End Timestamp Max: {flow_df['end_timestamp'].max()}, Min: {flow_df['end_timestamp'].min()}")
    
    # 过滤出位于特定时间段的流
    flow_df = flow_df[(flow_df['start_timestamp'] >= start_time) & (flow_df['end_timestamp'] <= end_time)]
    print(f"Final Flow Length: {len(flow_df)}")
    
    # 保留最多3万条数据
    if len(flow_df) > max_flows:
        flow_df = flow_df.sample(max_flows)
    
    # 按开始时间排序
    flow_df.sort_values(by=['start_timestamp'], inplace=True)
    
    # 保存过滤后的流数据到 CSV 文件
    flow_df.to_csv(filtered_flow_file, index=False)
    print(f"Filtered flow data saved to {filtered_flow_file}")

def load_and_filter_packets(packets_folder, start_time, end_time, filtered_packet_file):
    """加载并过滤包数据，保存到 CSV 文件"""
    print("Loading and filtering packet data...")
    # 读取Packet文件
    packet_files = [
        os.path.join(packets_folder, f)
        for f in os.listdir(packets_folder)
        if f.startswith('packets_') and f.endswith('.csv')
    ]
    
    packet_df = pd.DataFrame()
    for file in tqdm(packet_files, desc="Loading packet files"):
        temp_df = pd.read_csv(file)
        packet_df = pd.concat([packet_df, temp_df], ignore_index=True)

    # 确保时间为浮点数类型
    packet_df['timestamp'] = packet_df['timestamp'].astype(float)
    
    # 过滤出位于特定时间段的包
    packet_df = packet_df[(packet_df['timestamp'] >= start_time) & (packet_df['timestamp'] <= end_time)]
    
    # 按时间戳排序
    packet_df.sort_values(by=['timestamp'], inplace=True)

    print(packet_df.head(10))
    # 保存过滤后的包数据到 CSV 文件
    print(f"Final Packet Length: {len(packet_df)}")
    packet_df.to_csv(filtered_packet_file, index=False)
    print(f"Filtered packet data saved to {filtered_packet_file}")
    
def main():
    # 定义时间范围
    start_time = 1499442960
    end_time = 1499444160
    print(f"start_time: {start_time}, end_time: {end_time}")

    # 加载并过滤流数据
    load_and_filter_flows('Flow/Fri-Aft', start_time, end_time, 'Dataset/Fri_Aft_filtered_flow_file.csv')
    
    # 加载并过滤包数据
    load_and_filter_packets('packetsdata_Fri', start_time, end_time, 'Dataset/Fri_Aft_filtered_packet_file.csv')
    
    print("Process completed!")

if __name__ == "__main__":
    main()