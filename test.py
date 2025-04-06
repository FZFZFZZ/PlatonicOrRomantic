from datetime import datetime, timedelta
import pandas as pd

df = pd.read_csv('Data/train.csv')

print(df['Label'].value_counts())