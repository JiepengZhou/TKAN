# 获取包的时序特征，是根据序列中包的特征来进行的，并对不足32时间步的进行前补0
import pandas as pd
import numpy as np
import os
import sys
'''
def calculate_global_features(df):
    """
    对整个文件中的所有包计算时序特征
    """
    # 按时间戳排序
    df = df.sort_values(by='timestamp')
    # 1. 基本时序特征
    df['inter_arrival_time'] = df['timestamp'].diff()  # 时间间隔
    # 2. 滑动窗口（窗口大小为 3）
    window_size = 3
    # 3. 滑动窗口均值（包长度）
    df['rolling_mean_length'] = df['length'].rolling(window=window_size).mean()
    # 4. 滑动窗口标准差（包长度）
    df['rolling_std_length'] = df['length'].rolling(window=window_size).std()
    # 5. 滑动窗口最大值（包长度）
    df['rolling_max_length'] = df['length'].rolling(window=window_size).max()
    # 6. 滑动窗口最小值（包长度）
    df['rolling_min_length'] = df['length'].rolling(window=window_size).min()
    # 处理缺失值和无穷值
    df = df.fillna(0)  # 将 NaN 替换为 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # 将 inf 替换为 NaN
    df.fillna(method='bfill', inplace=True)  # 用后向填充处理剩余的 NaN
    return df
'''

def calculate_time_series_features(df, TS_columns):
    """
    计算每个 flow_id 内部的时序特征。
    """
    results = []
    for flow_id, group in df.groupby("flow_id"):
        group = group.sort_values(by='timestamp').copy()
        group.loc[:, 'inter_arrival_time'] = group['timestamp'].diff().fillna(0)
        
        # 计算滑动窗口特征
        window_size = 3
        group.loc[:, 'rolling_mean_length'] = group['length'].rolling(window=window_size).mean().fillna(0)
        group.loc[:, 'rolling_std_length'] = group['length'].rolling(window=window_size).std().fillna(0)
        group.loc[:, 'rolling_max_length'] = group['length'].rolling(window=window_size).max().fillna(0)
        group.loc[:, 'rolling_min_length'] = group['length'].rolling(window=window_size).min().fillna(0)
        
        
        # 你要干两次，一次是没有补0，单独包特征，一次是后补0，，这个要作为第一个图层
        # 截取或补齐数据到 32 个包
        if len(group) > 32:
            group = group.iloc[:32]  # 直接截断
        '''
        else:
            padding_rows = 32 - len(group)
            if padding_rows > 0:  # **防止空列表**
                padding = group.head(1).copy()
                if not padding.empty:
                    padding = padding.assign(**{col: 0 for col in TS_columns})
                    padding = pd.concat([padding] * padding_rows, ignore_index=True)
                    group = pd.concat([group, padding], ignore_index=True)  # **后补 0**
        '''
        results.append(group)
    
    return pd.concat(results, ignore_index=True)


def process_global_flows(input_file, output_file, target_columns, TS_columns):
    """
    处理文件中的所有包，计算全局时序特征并保存结果
    """
    df = pd.read_csv(input_file)

    result = calculate_time_series_features(df, TS_columns)
    result = result[target_columns]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result.to_csv(output_file, index=False)
    print(f"计算结果已保存到 {output_file}")

def main():
    input_file = sys.argv[1]  # 输入文件路径
    output_file = "TSF/" + sys.argv[2]  # 输出文件路径
    target_columns = [
        'src_ip', 'dst_ip', 'src_port', 'dst_port', 'flow_id', 'Label', 'pacc',
        'timestamp', 'length', 'inter_arrival_time',
        'rolling_mean_length', 'rolling_std_length', 'rolling_max_length', 'rolling_min_length',
    ]
    TS_columns = [
        'timestamp', 'length', 'inter_arrival_time',
        'rolling_mean_length', 'rolling_std_length', 'rolling_max_length', 'rolling_min_length',
    ]
    process_global_flows(input_file, output_file, target_columns, TS_columns)
    print("Done!")

if __name__ == "__main__":
    main()