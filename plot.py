# %%
# import plotly
# import plotly.express as px
import json, os, string
import numpy as np
import pandas as pd

# %%
def get_stats(path):
    if os.path.isfile(path):
        stats_m = json.load(open(fp_m, 'r'))
        return list(stats_m.values())
    if os.path.isdir(path):
        stat_fps = [os.path.join(path, v) for v in os.listdir(path)]
        return [
            json.load(open(_fp, 'r'))
            for _fp in stat_fps
        ]
    return None

# %%
fp_m = './logs/stats_master.json'
df_mr = pd.DataFrame(get_stats(fp_m))
df_mr

_s = {}
for _stat in df_mr.to_dict('records'):
    _time_stamp = _stat['telem']['time_stamp']
    _completed = _stat['telem']['completed']
    _r = {}
    _has_run = False
    for _key in ['train', 'val']:
        _sample_count = _stat['telem'].get('sample_count_{}'.format(_key))
        _r[_key] = {
            'sample_count': _sample_count,
            'vram': -1.,
        }
        if _key not in _stat:
            continue
        _acc = 0.0
        _loss = -1
        _best_epoch = 0
        _time_costs = []
        _vram = -1.
        for v in _stat[_key]:
            _has_run = True
            if v['acc'] > _acc:
                _acc = v['acc']
                _best_epoch = v['epoch']
            _loss = max(_loss, v['loss'])
            _time_costs.append(v['time_cost'])
            _vram = max(_vram, v['vram'])
        _time_cost = None
        _speed = None
        if len(_time_costs) >= 2:
            _time_cost = float(np.mean(_time_costs[1:]))
        elif len(_time_costs) == 1:
            _time_cost = float(_time_costs[0])
        if _time_cost is not None:
            _speed = _sample_count / _time_cost
        _r[_key].update({
            'acc': _acc,
            'loss': _loss,
            'time_cost': _time_cost,
            'best_epoch': _best_epoch,
            'sample_count': _sample_count,
            'speed': _speed,
            'vram': _vram,
        })
    
    if _has_run is False:
        continue
    _d = {
        'time_stamp': _time_stamp,
        'dataset': _stat['info']['dataset'],
        'bs': _stat['info']['bs'],
        'arch': _stat['info']['arch'],
        'epoch': _stat['info']['epoch'],
        'sample_count_train': _stat['telem']['sample_count_train'],
        'sample_count_val': _stat['telem']['sample_count_val'],
        'lr': _stat['info']['lr'],
        'completed': _completed,
        **{
            '{}.{}'.format(k, k2): v2
            for k, v in _r.items()
            for k2, v2 in v.items()
        },
    }
    _fingerprint = (_stat['info']['dataset'], _stat['info']['arch'], _stat['info']['bs'], _stat['telem']['time_stamp'])
    if _fingerprint in _s:
        if _completed and _time_stamp > _s[_fingerprint]['time_stamp']:
            _s[_fingerprint] = _d
    else:
        _s[_fingerprint] = _d


df_m = pd.DataFrame(_s.values())
# df_m = df_m[df_m['completed']].copy(True)
df_m


df_p = df_m[[
    'dataset', 'arch',
    # 'epoch',
    'bs',
    'lr',
    # 'completed',
    *[
        '{}.{}'.format(k, k2)
        for k2 in [
            # 'sample_count',
            'acc',
            'time_cost',
            'speed',
            # 'vram',
        ]
        for k in ['train', 'val']
    ],
]].sort_values(['dataset', 'val.acc', 'arch', 'bs'], ascending=False).copy(True)
# df_p = df_p
df_p

# %%