import pandas as pd
import numpy as np

df = pd.DataFrame([
           ['green', 'M', 10.1, 'class1'],
           ['red', 'L', 13.5, 'class2'],
           ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']

size_mapping = {
                'XL': 3,
                'L': 2,
                'M': 1}
df['size'] = df['size'].map(size_mapping)

print("\n{}".format(df))

inv_size_mapping = {v: k for k, v in size_mapping.items()}
print("\n{}".format(df['size'].map(inv_size_mapping)))

class_mapping = {label:idx for idx,label in
                 enumerate(np.unique(df['classlabel']))}
print("\n{}".format(class_mapping))

df['classlabel'] = df['classlabel'].map(class_mapping)
print("\n{}".format(df))