import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                          'ml/machine-learning-databases/'
                          'wine/wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol',
                    'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium',
                    'Total phenols', 'Flavanoids',
                    'Nonflavanoid phenols',
                    'Proanthocyanins',
                    'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines',
                    'Proline']

print('Class labels', np.unique(df_wine['Class label']))
print("\n{}".format(df_wine.head()))

# A convenient way to randomly partition this dataset into separate
# test and training(30%) datasets is to use the train_test_split function
# from scikit-learn's model_selection submodule:
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =\
     train_test_split(X, y,
                      test_size=0.3,
                      random_state=0,
                      stratify=y)

# The most commonly used splits are 60:40, 70:30, or 80:20,
# depending on the size of the initial dataset. 
# However, for large datasets, 90:10 or 99:1 splits into
# training and test subsets are also common and appropriate.