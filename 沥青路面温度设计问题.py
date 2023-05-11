# 导入相关模块
import numpy as np
from scipy.interpolate import interp1d

# 定义材料参数，分别为密度、比热容和导热系数
density = np.array([2100, 1800, 1600, 1500])
heat_capacity = np.array([900, 810, 810, 880])
t_c = np.array([4680, 3888, 4392, 4392])

# 定义结构层厚度
thickness = np.array([0.12, 0.18, 0.18, 0.8])
# 定义每层迭代次数
depth = np.array([10, 10, 10, 5])
# 定义类别
catefory = np.array([0, 1, 2, 3, 4])

# 定义时间点
time = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
# 沥青表面温度
y = np.array([31.2, 29.2, 28.4, 27.7, 28.7, 31.9, 33.9, 36.0, 37.2, 35.5, 34.0, 32.1])
# 土层中间温度
y2 = np.array([29.0, 28.9, 28.9, 28.8, 28.6, 28.4, 28.2, 28.4, 28.6, 28.8, 28.9, 29.0])

# 插值拟合
xp2 = np.linspace(0, 22, 86400)  # 生成用于绘制插值曲线的 x 坐标
f = interp1d(time, y, kind='cubic')  # 'cubic' 表示使用三次样条插值
yp2 = f(xp2)
f2 = interp1d(time, y2, kind='cubic')
yp3 = f2(xp2)

# 定义初始温度分布
T = np.zeros((4, 11, len(xp2))) # T表示温度分布的二维矩阵，第一维表示位置，第二位表示时间
tp = np.linspace(28.9, 29.2, 44)
tp = tp[::-1]
tp = tp.reshape(4, 11)
T[:, :, 7200] = tp # 2点初始化为给定的值
T[3, 10, :] = 26
T[0, 0, :] = yp2
T[3, 5, :] = yp3
dt = 1
dx = [i / 10 for i in thickness]
alpha = t_c / (density * heat_capacity) # 计算热扩散系数
alpha /= 3600 # 热传导率中的单位度量转换

# 迭代求解温度
for j in range(7200, len(xp2) * 8, 1):
    j = (int)(j%len(xp2))
    # 处理内部情况
    for k in range(4):
        for i in range(1, depth[k], 1):
            T[k][i][(int)((j+1)%len(xp2))] = T[k][i][j] + alpha[k] * dt / (dx[k]**2) * (T[k][i-1][j] - 2 * T[k][i][j] + T[k][i+1][j])
    # 处理边界情况
    j = (int)((j+1)%len(xp2))
    da = t_c[0] / dx[0]
    db = t_c[1] / dx[1]
    T[0][depth[0]][j] = (da * T[0][depth[0] - 1][j] + db * T[1][1][j]) / (da+db)
    T[1][0][j] = T[0][depth[0]][j]
    da = t_c[1] / dx[1]
    db = t_c[2] / dx[2]
    T[1][depth[1]][j] = (da * T[1][depth[1] - 1][j] + db * T[2][1][j]) / (da+db)
    T[2][0][j] = T[1][depth[1]][j]
    da = t_c[2] / dx[2]
    db = t_c[3] / dx[3]
    T[2][depth[2]][j] = (da * T[2][depth[2] - 1][j] + db * T[3][1][j]) / (da+db)
    T[3][0][j] = T[2][depth[2]][j]

# 输出结果
for k in range(1, 4):
    print('The temperature at the interface between layer', k, 'and layer', k + 1, 'at each whole point in time is:') # 输出提示信息
    print(T[k][0][0::7200]) # 输出每个时间点上第i+1行

# 绘图部分
import matplotlib.pyplot as plt

xx = [i for i in range(24)]
plt.plot(T[1][0][0::3600], color = 'blue', marker = 'o', markersize = 3, linestyle = 'solid', linewidth = 1.5, label = '混凝土层与基层交界面温度')
plt.plot(T[2][0][0::3600], color = 'orange', marker = 's', markersize = 3, linestyle = 'dashed', linewidth = 1.5, label = '基层与底层交界面温度')
plt.plot(T[3][0][0::3600], color = 'r', marker = 'd', markersize = 3, linestyle = 'dashdot', linewidth = 1.5, label = '底层与土层交界面温度')
plt.plot(yp2[0::3600], color = 'black', marker = '<', markersize = 3, linestyle = 'dotted', linewidth = 1.5, label = '沥青表面的温度')
plt.plot(T[3][10][0::3600], color = 'g', marker = '>', markersize = 3, linestyle = 'dotted', linewidth = 1.5, label = '土层下方交界面温度')

plt.xlabel('time step / h')
plt.xticks(xx)
plt.title('temperature at the soil layer interface')
plt.ylabel('temperature / ℃')
plt.legend(loc = 'upper left', fontsize = 'x-small')

plt.show()

# 结果保存到Excel文件中
import pandas as pd

data = {'时刻': ['0:00' , '1:00' , '2:00' , '3:00' , '4:00' , '5:00' , '6:00' , '7:00' , '8:00' , '9:00' , '10:00', '11:00',
                '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
        '沥青表面温度(℃)':yp2[0::3600],
        '混凝土层与基层交界面温度(℃)':T[1][0][0::3600],
        '基层与底层交界面温度(℃)':T[2][0][0::3600],
        '底层与土层交界面温度(℃)':T[3][0][0::3600],
        '土层下方交界面温度(℃)':T[3][10][0::3600],
        }

df = pd.DataFrame(data)

# 将DataFrame写入Excel文件
writer = pd.ExcelWriter('temperature.xlsx')
df.to_excel(writer, index=False)
writer.save()