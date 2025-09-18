import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

packets_folder = 'packetsdata_Fri'

print("Loading packet data...")
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

# 打印出包的时间范围
print(f"Packet_Timestamp_MAX (datetime): {packet_df['timestamp'].max()}, Packet_Timestamp_MIN (datetime): {packet_df['timestamp'].min()}")