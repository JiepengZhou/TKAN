import pandas as pd
import sys

# 读取 CSV 文件
file_name = sys.argv[1]  # 替换为你的文件路径
df = pd.read_csv(file_name)

# 检查每行是否有 9 列
incorrect_rows = df[df.apply(lambda row: len(row) != 9, axis=1)]

# 如果有不正确的行，则输出这些行的索引
if not incorrect_rows.empty:
    print(f"以下行有错误，列数不为 9:")
    print(incorrect_rows)
else:
    # 统计每个 flow_id 的实际行数
    flow_group_counts = df.groupby('flow_id').size()

    # 检查是否每个 flow_id 的行数为 32
    incorrect_flow_ids = flow_group_counts[flow_group_counts != 32]

    # 输出不符合条件的 flow_id
    if not incorrect_flow_ids.empty:
        print("以下 flow_id 的行数不为 32:")
        print(incorrect_flow_ids)
    else:
        print("所有 flow_id 的行数均为 32，且每行都有 9 列。")
