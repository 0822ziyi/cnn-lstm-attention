from scipy.stats import pearsonr
import pandas as pd


test_results = pd.read_csv('test_results.txt', sep='\t')


true_values = test_results['True'].values
pred_values = test_results['Pred'].values


correlation, _ = pearsonr(true_values, pred_values)

print(f'Pearson correlation coefficient: {correlation:.4f}')
