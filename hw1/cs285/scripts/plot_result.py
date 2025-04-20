import numpy as np
import matplotlib.pyplot as plt

# behavior cloning data 
lr = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2]
eval_ret_mean = np.array([-322.386, 320.736, 3637.338, 4604.182, 4609.212, 526.881])
eval_ret_std = np.array([479.651, 346.126, 330.062, 89.287, 37.984, 226.145])
halfcheetah_ret_mean = np.array([316.509, 1367.904, 3520.899, 3919.686, 3821.584,  2347.498])
halfcheetah_ret_std = np.array([94.034, 181.138, 112.912, 121.531, 120.213, 137.518])

# dagger data
# eval_batch_size 5000
# n_iter = [2, 3, 5, 6, 7, 8, 9, 10, 15, 20]
# eval_ret_mean = np.array([4685.809, 4669.874, 4668.167, 4737.813, 4405.421, 4722.252, 4796.916 ,4549.032, 4755.949, 4452.156])
# eval_ret_std = np.array([166.117, 115.880, 96.770, 141.013, 964.862, 105.985, 75.279 ,84.578, 68.399, 343.086])
# halfcheetah_eval_ret_mean = np.array([ 3898.699, 4084.831, 4013.010, 4018.018, 4050.052, 3988.786, 4023.326, 4068.218, 4052.558, 4093.124])
# halfcheetah_eval_ret_std = np.array([89.590, 47.203, 101.588, 54.033,  51.706, 41.148, 113.091, 86.083, 39.023, 46.044])
# # eval_batch_size 50000
# n_iter = [10, 20, 30, 40, 50]
# eval_ret_mean = np.array([246.504])
# eval_ret_std = np.array([79.773])

# 绘制原始数据、均值和方差区域
# plt.figure(figsize=(10, 6))
# plt.plot(n_iter, halfcheetah_eval_ret_mean, label='Mean', alpha=0.5, color='gray', marker='o')
# plt.fill_between(
#     n_iter,
#     halfcheetah_eval_ret_mean - halfcheetah_eval_ret_std,  # 均值 - 标准差（sqrt(方差)）
#     halfcheetah_eval_ret_mean + halfcheetah_eval_ret_std,  # 均值 + 标准差
#     color='blue', alpha=0.1, label='Mean ± Std Dev'
# )
# plt.xlabel('n_iter')

plt.plot(lr, halfcheetah_ret_mean, label='Mean', alpha=0.5, color='gray', marker='o')
plt.fill_between(
    lr,
    halfcheetah_ret_mean - halfcheetah_ret_std,  # 均值 - 标准差（sqrt(方差)）
    halfcheetah_ret_mean + halfcheetah_ret_std,  # 均值 + 标准差
    color='blue', alpha=0.1, label='Mean ± Std Dev'
)
plt.xscale('log')
plt.xlabel('lr')

plt.ylabel('return')
plt.title('Performance of HalfCheetah task(5 rollouts)')
plt.legend()
plt.grid(True)
plt.show()