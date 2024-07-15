import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
import numpy as np


valid_results = pd.read_csv('test_results.txt', sep='\t')
true_values = valid_results['True']
predicted_values = valid_results['Pred']


plt.figure(figsize=(10, 6))
plt.scatter(true_values, predicted_values, alpha=0.3)
plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values on Validation Set')
plt.show()


mse = mean_squared_error(true_values, predicted_values)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true_values, predicted_values)
r2 = r2_score(true_values, predicted_values)


shifted_true_values = true_values - true_values.min() + 1
shifted_predicted_values = predicted_values - predicted_values.min() + 1
msle = mean_squared_log_error(shifted_true_values, shifted_predicted_values)


print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R²: {r2}')
print(f'MSLE: {msle}')
