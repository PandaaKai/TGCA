#!/bin/bash

# 定义数据目录
data_dir="data"

# 检查数据目录是否存在
if [ ! -d "$data_dir" ]; then
    echo "Data directory does not exist."
    exit 1
fi

# 遍历 data 目录下的所有子目录
for relative_data_dir in "$data_dir"/*; do
    if [ -d "$relative_data_dir" ]; then
        full_data_prefix="$data_dir/$(basename "$relative_data_dir")"
        echo "Processing directory: $full_data_prefix"

        # 执行 Python 脚本
        python main.py --data_prefix "$full_data_prefix" --train_config "./train_config/config.yml" --repeat_time 1
    fi
done

echo "All directories processed."
read -p "Press any key to continue..."