# %%
# import plotly
# import plotly.express as px
import json, os, string
import numpy as np
import pandas as pd

# %%
dp = 'logs/stats'
stat_fps = [os.path.join(dp, v) for v in os.listdir(dp)]
stats = {
    _fp: json.load(open(_fp, 'r'))
    for _fp in stat_fps
}
len(stats)

# %%
df = pd.DataFrame([
    {
        'fp': _fp,
        'info': d['info'],
        'telem': d['telem'],
        'train': [v for v in d['train'][1:]]
    }
    for _fp, d in stats.items()
])

# %%
fp_m = './logs/stats_master.json'
stats_m = json.load(open(fp_m, 'r'))
df_mr = pd.DataFrame(list(stats_m.values()))
df_mr

# %%
_s = []
for _stat in df_mr.to_dict('records'):
    _completed = _stat['telem']['completed']
    _r = {}
    if _completed:
        for _key in ['train', 'val']:
            if _key not in _stat:
                continue
            _acc = 0.0
            _loss = 99.9999
            _best_epoch = 0
            _time_costs = []
            for v in _stat[_key]:
                if v['acc'] > _acc:
                    _acc = v['acc']
                    _best_epoch = v['epoch']
                _loss = max(_loss, v['loss'])
                _time_costs.append(v['time_cost'])
            _time_cost = None
            if len(_time_costs) >= 2:
                _time_cost = float(np.mean(_time_costs[1:]))
            elif len(_time_costs) == 1:
                _time_cost = float(_time_costs[0])
            _r[_key] = {
                'acc': _acc,
                'loss': _loss,
                'time_cost': _time_cost,
                'best_epoch': _best_epoch,
                'sample_count': _stat['telem'].get('sample_count_{}'.format(_key))
            }
    _s.append({
        'dataset': _stat['info']['dataset'],
        'bs': _stat['info']['bs'],
        'arch': _stat['info']['arch'],
        'epoch': _stat['info']['epoch'],
        'sample_count_train': _stat['telem']['sample_count_train'],
        'sample_count_val': _stat['telem']['sample_count_val'],
        'completed': _completed,
        **{
            '{}.{}'.format(k, k2): v2
            for k, v in _r.items()
            for k2, v2 in v.items()
        },
    })


df_m = pd.DataFrame(_s)

# %%