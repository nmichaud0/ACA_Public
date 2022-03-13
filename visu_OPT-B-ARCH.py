import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/Users/nizarmichaud/PycharmProjects/ACA_Public/B_ARCH_output/combination_eval_1_2_3.csv'
df = pd.read_csv(path)

voting = df['voting'].tolist()
all_bacc = df['all_bacc'].tolist()
model = df['combination'].tolist()
size = df['n'].to_numpy()

combination_bacc = df['combination_bacc'].tolist()

colors = df['voting'].astype('category').cat.codes


new_comb_bacc = []
comb_bacc_mean = []
for i in combination_bacc:
    i = i.removeprefix('[')
    i = i.removesuffix(']')

    arr = np.fromstring(i, sep=',')

    new_comb_bacc.append(arr)
    comb_bacc_mean.append(np.mean(arr))

"""
fig = go.Figure(data=go.Splom(dimensions=[dict(values=all_bacc, label='all balanced accuracy'),
                                          dict(values=comb_bacc_mean, label='combination balanced accuracy mean')],
                              text=[df['combination'], voting], marker=dict(color=colors, showscale=False)
                              )
                )


fig = go.Figure(data=go.Scatter(x=all_bacc, y=comb_bacc_mean,
                                mode='markers'))
                                #marker=dict(color=colors, size=size),
                                #text=model))
fig.show()

plt.scatter(x=all_bacc, y=comb_bacc_mean, c=colors, s=size)

plt.show()"""

# TODO: ???? MPL & Plotly are only showing 4 dots when plotting... excel seems to have done something weird with the
#  data, the xlsx is ~200Kb and the csv is ~10Mb...
