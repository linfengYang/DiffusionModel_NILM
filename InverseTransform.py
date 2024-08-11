import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

applianceName='Washingmachine'
df = pd.read_csv(f'home2_{applianceName}.csv')
Ftime_series=df['power'].values.reshape(-1, 1)
# Ftime_series[Ftime_series < 2] = 0

scaler = MinMaxScaler()
normalized_series = scaler.fit_transform(Ftime_series)
df = pd.read_csv(f'generatedData/MinMaxgenerateWashMachineH2_dataHalf.csv')
flattened_data = df['power'].values.reshape(-1, 1)
inverse_normalized_data = scaler.inverse_transform(flattened_data)
generate_data_restored = inverse_normalized_data

arr1 = generate_data_restored.flatten()
arr2 = Ftime_series.flatten()
fig, axs = plt.subplots(2, figsize=(8, 8))
axs[0].plot(arr1[:20000], linestyle='-', color='b', label='Array 1')
axs[0].set_title(f'generate {applianceName}')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('power')
axs[0].legend()
axs[1].plot(arr2[:20000], linestyle='-', color='r', label='Array 2')
axs[1].set_title(f'origin {applianceName}')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('power')
axs[1].legend()

plt.tight_layout()
plt.show()
flattened_data=arr1
df = pd.DataFrame(flattened_data, columns=['power'])
df.to_csv(f'generatedData/{applianceName}_dataH2.csv', index=False)