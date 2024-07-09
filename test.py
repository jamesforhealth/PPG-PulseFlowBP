import matplotlib.pyplot as plt
import numpy as np

# 生成隨機分佈的點
mean = 0
std_dev = 1
num_points = 1000

x = np.random.normal(mean, std_dev, num_points)
y = np.random.normal(mean, std_dev, num_points)

# 繪製散佈圖
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.title('Scatter Plot of 1000 Random Points with Normal Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()