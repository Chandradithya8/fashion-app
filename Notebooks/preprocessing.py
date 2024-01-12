import pandas as pd
import os
import numpy as np
folder_path = "images"
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

df = pd.read_csv("styles.csv", on_bad_lines="skip")
current = []
for x in df.id.values:
    if str(x)+".jpg" not in file_names:
        current.append(False)
    else:
        current.append(True)
df["Good"] = np.array(current)


idx = []
category = []
color = []
for a, b, c, d in zip(df.id.values, df.masterCategory.values, df.baseColour.values, df.Good.values):
    if d:
        idx.append(a)
        category.append(b)
        color.append(c)


data = pd.DataFrame()
data["id"] = np.array(idx)
data["masterCategory"] = np.array(category)
data["baseColour"] = np.array(color)

data.to_csv("preprocessed.csv", index = False)
