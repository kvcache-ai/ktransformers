import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
file_path = 'data.txt'  # 替换为你的文件路径
df = pd.read_csv(file_path, sep=r'\s+', names=['m', 'n', 'tflops'])

# 创建数据透视表，行为 m，列为 n，值为 tflops
pivot_table = df.pivot_table(index='m', columns='n', values='tflops')

# 画热力图
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='viridis')
plt.title('TFLOPS Heatmap')
plt.xlabel('n')
plt.ylabel('m')
plt.tight_layout()
plt.show()
