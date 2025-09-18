# 第二步，修改flow文件的时间戳为UNIX时间戳
import pandas as pd

flow_df = pd.read_csv('CICIDS/CICIDS2017/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
# 去除烦人的空格
flow_df.columns = flow_df.columns.str.strip()
print(flow_df.head(10))
# 只保留 Label 为 "DDos" 的数据
flow_df = flow_df[flow_df['Label'] == 'DDoS']
print(f"DDoS Len: {len(flow_df)}") 
# 修改字符串类型为datetime类型
flow_df['Timestamp'] = pd.to_datetime(flow_df['Timestamp'], format='%d/%m/%Y %H:%M')
# 打印 datetime 类型的最大值和最小值
# print(flow_df['Timestamp'].isna().sum()) # 检查是否有nan值
# print(flow_df['Timestamp'].isna().sum()) # 查看nat值的个数
print(f"Flow_df_Timestamp_MAX (datetime): {flow_df['Timestamp'].max()}, Flow_df_Timestamp_MIN (datetime): {flow_df['Timestamp'].min()}")

# +8个小时去掉时区再转为UNIX时间戳
flow_df['Timestamp'] = flow_df['Timestamp'] + pd.Timedelta(hours=12)
flow_df['Timestamp'] = flow_df['Timestamp'].astype('int64') // 10**9
# 打印出当前时间戳的最大值和最小值
print(f"Flow_df_Timestamp_MAX : {flow_df['Timestamp'].max()}, FLow_df_Timestamp_MIN: {flow_df['Timestamp'].min()}")

#重新写入文件
# 按每 10,000 行拆分保存
chunk_size = 10_000
for i, chunk in enumerate(range(0, len(flow_df), chunk_size)):
    chunk_df = flow_df.iloc[chunk:chunk + chunk_size]
    filename = f"Flow/Fri-Aft/Flow_DDos_Part_{i+1}.csv"
    chunk_df.to_csv(filename, index=False)
    print(f"Saved {filename}")
print("Success!")