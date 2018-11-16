import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
,,,
5.0,6.0,,8.0
10.0,11.0,12.0,'''
# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))
print("csv-filen som vi läst in\n{}".format(df))

# only drop rows where all columns are NaN
# (returns the whole array here since we don't
# have a row with where all values are NaN
print("\nTar bort hela raden om alla värden är NaN.\n{}".format(df.dropna(how='all')))

# drop rows that have less than 4 real values
print("\nTa bort alla rader som inte har 4 riktiga värden\n{}".format(df.dropna(thresh=4)))

# only drop rows where NaN appear in specific columns (here: 'C')
print("\nTa bort NaN i kolumnn C\n{}".format(df.dropna(subset=['C'])))

df.isnull().sum()

# mean imputation. Där det saknas ett värde räkna ut ett troligt värde
imr = Imputer(missing_values='NaN', strategy='mean', axis=1)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print("\n{}".format(imputed_data))