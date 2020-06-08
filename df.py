import pandas as pd
p={}
p['fp16'] = 1000
p['fp32'] = 2.33
df = pd.DataFrame.from_dict(p, orient='index', columns=["Total runtime"])
df["Average runtime"] = df["Total runtime"]/1000
print(df)
