# %%
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json, time, os

# %%
fp = '/home/researcher/VT/ViT_torch/object_detr/logs/stats_210802_162009.json'
fp = '/home/researcher/VT/ViT_torch/object_detr/logs/stats_210802_165201.json'
_log = json.load(open(fp, 'r'))
bbox_key_prefix = 'coco_eval_bbox.ap'
bbox_key_prefix_len = len(bbox_key_prefix) - 2
df = pd.DataFrame([
    {
        **{
            k: d[k]
            for k in ['epoch', 'time_train', 'time_val']
        },
        **{
            '{}.{}'.format(_split, k): d[_split][k]
            for _split in ['train', 'val']
            for k in ['loss', 'class_error']
        },
        **{
            k[bbox_key_prefix_len : ]: d['val'][k]
            for k in d['val'].keys()
            if k.startswith(bbox_key_prefix)
        },
    }
    for d in _log['logs']
])
df

# %%