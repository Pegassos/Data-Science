n, p = [int(x) for x in input().split()]
X = []
for i in range(n):
  X.append([float(x) for x in input().split()])

### -------------------
import numpy as np
import pandas as pd

lst = [float(x) if x != 'nan' else np.NaN for x in input().split()]

df = pd.DataFrame(lst)
df = df.fillna(df.mean().round(1))

print(df[0])

# print('dtype:', df.to_numpy().dtype)


