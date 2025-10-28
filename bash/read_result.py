import os
from glob import glob

# 设置根目录
base_dir = "/opt/data/private/FDAARC/checkpoints/FAGes/SSL/UDA_our_T=user3_k=40_06110630"

# 获取所有子目录中的 eval/results.txt 路径
results_files = glob(os.path.join(base_dir, "C_user3_fsl*/eval/results.txt"))

srcacc = []
tgtacc = []

# 遍历每个文件，提取第二行和第三行（源域和目标域准确率）
for file_path in results_files:
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            if len(lines) >= 3:
                src_line = lines[1].strip()
                tgt_line = lines[2].strip()

                # 提取百分数中的数字，例如 "源域准确率: 95.67%" → 95.67
                src_val = float(''.join(filter(lambda c: (c.isdigit() or c == '.'), src_line)))
                tgt_val = float(''.join(filter(lambda c: (c.isdigit() or c == '.'), tgt_line)))

                srcacc.append(src_val)
                tgtacc.append(tgt_val)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


# 构造字符串：=AVERAGE(95.67,94.21,96.88)
src_average_formula = f"=AVERAGE({','.join([f'{x:.2f}' for x in srcacc])})"
tgt_average_formula = f"=AVERAGE({','.join([f'{x:.2f}' for x in tgtacc])})"

print(src_average_formula)
print(tgt_average_formula)
