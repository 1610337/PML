import pandas as pd


df = pd.DataFrame(pd.DataFrame(), columns= ["i", "i+10"])

for i in range(0,10):
    print(i, i+10)
    df.loc[i] = [i, i+10]

print(df)