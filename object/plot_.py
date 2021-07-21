# %%
import json, time, os
import pandas as pd
import numpy as np
import plotly.express as px

# %%
fps = [
    # '/home/researcher/VT/object_pipeline/logs/stats_210719_201752.json',
    # '/home/researcher/VT/object_pipeline/logs/stats_210720_091930.json',
    '/home/researcher/VT/object_pipeline/logs/stats_210720_193141.json',
]
data = [json.load(open(fp, 'r')) for fp in fps]
data2 = []
for i, d in enumerate(data[0]['logs']):
    d2 = {
        'epoch': i + 1,
        **d['train'],
        **{
            '{}.{}'.format(k, k2): v2
            for k, v in d['val'].items()
            for k2, v2 in v.items()
        },
    }
    data2.append(d2)

df = pd.DataFrame(data2)
df

# %%
px.line(
    df,
    x='epoch',
    # y=[v for v in df.columns if v not in ['epoch'] and v.startswith('bbox.ap')],
    # y=[v for v in df.columns if v not in ['epoch'] and (v.startswith('bbox.ap') or v.startswith('loss'))],
    y=['loss', 'bbox.ap', 'bbox.ap50', 'bbox.ap75'],
    range_y=[0., 1.],
    width=1200,
    height=800,
)


# %%

