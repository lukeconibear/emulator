"""
Maxi-min Latin hypercube spacefilling design of simulator runs
For each of the five features, designed over 100,000 iterations
"""

import pandas as pd
from pyDOE import lhs
import seaborn as sns

design = lhs(5, samples=50, criterion='maximin', iterations=100000) * 1.5

df = pd.DataFrame(data=design, columns=['RES', 'IND', 'TRA', 'AGR', 'ENE'])
df.to_csv('latin_hypercube_inputs.csv')

df.describe()

sns.pairplot(df, diag_kind='hist')
plt.show()