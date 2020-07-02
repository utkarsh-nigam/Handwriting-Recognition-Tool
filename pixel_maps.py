import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


data = pd.read_csv("final_data_a_z.csv")

print(data["label"].value_counts())

graph_data=pd.DataFrame()
for i in range(0,26):
    tempdf=data[data["label"]==i]
    tempdf=tempdf.sample(n=1120)
    graph_data=graph_data.append(tempdf)

graph_data=graph_data.drop(columns=["label"])

unique_values=graph_data.nunique().tolist()

non_empty=graph_data.isin([0]).sum(axis=0).tolist()

uniquevalues=np.array(unique_values).reshape(28,28)

nonempty=np.abs(29120-np.array(non_empty).reshape(28,28))

focus_unique=np.delete(nonempty,[0,1,2],0)
focus_unique=np.delete(focus_unique,[0,1,2],1)
focus_unique=np.delete(focus_unique,[22,23,24],1)
focus_unique=np.delete(focus_unique,[22,23,24],0)

focus_empty=np.delete(nonempty,[0,1,2],0)
focus_empty=np.delete(focus_empty,[0,1,2],1)
focus_empty=np.delete(focus_empty,[22,23,24],1)
focus_empty=np.delete(focus_empty,[22,23,24],0)


sns.heatmap(focus_unique-1)
plt.axis('off')
plt.show()

sns.heatmap(nonempty,cmap="Blues",cbar=False)
plt.axis('off')
plt.show()
