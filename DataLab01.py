import pandas as pd

df = pd.read_csv("./diabetes.csv")

print(df.head())
df.head().to_html("./ch5-2-1.html")
print(df.shape)