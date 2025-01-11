import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# 定義低通濾波器
def lowpass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# 生成一個示例信號
fs = 100  # 采樣頻率
t = np.linspace(0, 10, 10 * fs, endpoint=False)  # 時間軸
signal = np.sin(1.2 * 2 * np.pi * t) + 1.5 * np.cos(9 * 2 * np.pi * t)

# 計算二階微分信號
second_derivative = np.gradient(np.gradient(signal))

# 應用二階低通濾波器模擬二次積分
cutoff = 0.5  # 截止頻率
filtered_signal = lowpass_filter(second_derivative, cutoff, fs)

diff = filtered_signal - signal
print(f'diff:{diff}, abs diff mean : {np.mean(np.abs(diff))}')
print(np.mean(np.abs(signal)))
# 繪圖
# plt.figure(figsize=(12, 6))
# plt.plot(t, signal, label='Original Signal')
# plt.plot(t, second_derivative, label='Second Derivative')
# plt.plot(t, filtered_signal, label='Filtered Signal (Simulated Second Integration)')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.title('Low-Pass Filter Simulating Second Integration')
# plt.show()