import pandas as pd
import matplotlib.pyplot as plt


csv1 = pd.read_csv('microwavemaeloss_data.csv')
csv2 = pd.read_csv('microwavemseloss_data.csv')
csv3 = pd.read_csv('microwaverourloss_data.csv')

first_col_csv1 = csv1.iloc[:, 0]
second_col_csv1 = csv1.iloc[:, 1]

first_col_csv2 = csv2.iloc[:, 0]
second_col_csv2 = csv2.iloc[:, 1]

first_col_csv3 = csv3.iloc[:, 0]
second_col_csv3 = csv3.iloc[:, 1]


plt.figure(figsize=(10, 6))
plt.plot(first_col_csv1, label='MAE loss')
plt.plot(first_col_csv2, label='MSE loss')
plt.plot(first_col_csv3, label='Our loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('MicrowaveTraining loss')
plt.legend(fontsize=15)
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(second_col_csv1, label='MAE loss')
plt.plot(second_col_csv2, label='MSE loss')
plt.plot(second_col_csv3, label='Our loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Second Column of Three CSV Files')
plt.legend()
plt.grid(True)
plt.show()
