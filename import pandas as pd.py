import pandas as pd
medquad = pd.read_csv("medquad.csv")

med_dict = medquad.to_dict(orient="records")
print(med_dict[:1])
