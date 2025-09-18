import pandas as pd
import sys

# 读取 CSV 文件
file_name = sys.argv[1]  # 输入的原始文件路径
output_file = sys.argv[2]  # 处理后的输出文件路径

# 读取数据
df = pd.read_csv(file_name)

# 统计每个 flow_id 的行数
flow_group_counts = df.groupby('flow_id').size()

# 过滤出行数为 32 的 flow_id
valid_flow_ids = flow_group_counts[flow_group_counts == 32].index

# 保留这些 flow_id 的数据
filtered_df = df[df['flow_id'].isin(valid_flow_ids)]

# 保存到新文件
filtered_df.to_csv(output_file, index=False)

print(f"处理完成，已保存 {len(valid_flow_ids)} 个有效 flow_id 到 {output_file}")
