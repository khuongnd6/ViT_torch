# %%
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json, time, os
import colorsys
from PIL import Image

# %%
fp = '/home/researcher/VT/ViT_torch/object_detr/logs/stats_210802_162009.json'
fp = '/home/researcher/VT/ViT_torch/object_detr/logs/stats_210802_165201.json'
fp = '/home/researcher/VT/ViT_torch/object_detr/logs/stats_210802_185821.json'
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
# all_labels_train = [
#     # 'background',
#     'airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench',
#     'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot',
#     'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut',
#     'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse',
#     'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange',
#     'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich',
#     'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign',
#     'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush',
#     'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']
# len(all_labels_train)

# _classes_dict = {
#     int(k): v
#     for k, v in json.load(open('/home/researcher/VT/ViT_torch/object_detr/coco_classes.json', 'r')).items()
# }
classes_list = []
for k, v in json.load(open('/home/researcher/VT/ViT_torch/object_detr/coco_classes.json', 'r')).items():
    _id = int(k)
    assert _id >= 0
    while _id >= len(classes_list):
        classes_list.append(None)
    classes_list[_id] = v

if len(classes_list) < 1:
    classes_list.append(None)
if classes_list[0] is None:
    classes_list[0] = 'background'
print(np.mean([isinstance(v, str) for v in classes_list]), '| [0, 1] (higher is better)')

# %%
root_dp = '/host/ubuntu/torch/coco2017/validation'
root_dp = '/host/ubuntu/torch/coco2017/train'
label_fp = os.path.join(root_dp, 'labels.json')

# %%
label_json = json.load(open(label_fp, 'r'))
len(label_json), type(label_json)

# %%
# label_json['images'][:5], label_json['annotations'][:5]

# anno_dict = {
#     anno['image_id']: anno
#     for anno in label_json['annotations']
# }
class CocoManager:
    def __init__(self, root_path=None, data_path=None, label_path=None, class_dict_path=None):
        if isinstance(root_path, str):
            self.data_path = os.path.join(root_path, 'data')
            self.label_path = os.path.join(root_path, 'labels.json')
        else:
            self.data_path = data_path
            self.label_path = label_path
        
        assert isinstance(self.data_path, str) and os.path.isdir(self.data_path)
        assert isinstance(self.label_path, str) and os.path.isfile(self.label_path)
        
        
        if isinstance(class_dict_path, str):
            with open(class_dict_path, 'r') as f:
                self.class_dict = {
                    **{
                        int(k): str(v)
                        for k, v in json.load(f).items()
                    },
                    0: 'background',
                }
        
        else:
            self.class_dict = {
                0: 'background',
            }
        
        self.class_list = []
        for k, v in self.class_dict.items():
            self.class_list
        # class_count counts the number of classes > 0
        self.class_count = len(self.class_dict) - 1
        



image_dict = {
    d['id']: {
        **d,
        'index': i,
        'anno': [],
        'has_anno': False,
        'contains_objs': [],
    }
    for i, d in enumerate(label_json['images'])
}
miss_count = 0
for anno in label_json['annotations']:
    image_id = anno['image_id']
    if image_id not in image_dict:
        miss_count += 1
        continue
    image_dict[image_id]['anno'].append(anno)
    image_dict[image_id]['has_anno'] = True
    cid = anno['category_id']
    if cid not in image_dict[image_id]['contains_objs']:
        image_dict[image_id]['contains_objs'].append(cid)

len(image_dict), np.mean([int(d['has_anno']) for d in image_dict.values()]), miss_count


# %%
image_list = sorted(
    [d for d in image_dict.values() if d['has_anno']],
    key=lambda v: v['id']
)
len(image_list)

# %%
def get_image(idx=0, classes=[]):
    if isinstance(classes, list):
        pass
    d = image_list[idx]
    fp = os.path.join(root_dp, 'data', d['file_name'])
    img = Image.open(fp)
    return img, d
    

# %%
def annotate(fp, df=None, model=None, max_size=1200):
    if isinstance(fp, Image.Image):
        img = fp
    else:
        img = Image.open(fp)
    
    img_size = list(img.size)
    assert max_size >= 2
    scale_factor = min([max_size / v for v in img_size])
    target_size = [v * scale_factor for v in img_size]
    # print(img_size, 'x', scale_factor, '=', target_size)
    fig = go.Figure()

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, target_size[0]],
            y=[0, target_size[1]],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, target_size[0]]
    )
    fig.update_yaxes(
        visible=False,
        range=[0, target_size[1]],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x",
        autorange="reversed",
    )
    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=target_size[0],
            y=0,
            sizey=-target_size[1],
            # y=target_size[1],
            # sizey=target_size[1],
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img,
            # source="https://raw.githubusercontent.com/michaelbabyn/plot_data/master/bridge.jpg",
        )
    )
    # Configure other layout
    fig.update_layout(
        width=target_size[0],
        height=target_size[1],
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    
    if df is None and model is not None:
        results = model([fp])
        df = results.pandas().xyxy[0]
    
    if isinstance(df, list):
        df = pd.DataFrame(df)
    df2 = df.copy(True)
    for k1, k2 in {
            'xmin': 'x0',
            'xmax': 'x1',
            'ymin': 'y0',
            'ymax': 'y1',
            }.items():
        if k1 in df2.columns:
            df2 = df2.rename(columns={k1: k2})
    for k0, k1, kr in [['x0', 'x1', 'w'], ['y0', 'y1', 'h']]:
        if kr in df2.columns:
            df2[k1] = df2[k0] + df2[kr]
    # if 'xmin' in df2.columns:
    #     df2 = df2.rename(columns={
    #         'xmin': 'x0',
    #         'xmax': 'x1',
    #         'ymin': 'y0',
    #         'ymax': 'y1',
    #     })
    
    for k in ['x0', 'x1', 'y0', 'y1']:
        df2[k] = df2[k] * scale_factor
    df2['text_x'] = (df2['x0'] + df2['x1']) / 2
    df2['text_y'] = df2['y0'] - 12
    
    _class_count = 80
    _classes = set(list(df2['class']))
    for i, _class in enumerate(sorted(list(_classes))):
        _rgb = '#' + ''.join([
            hex(int(np.clip(v * 255, 0, 255)))[2:][-2:].rjust(2, '0')
            for v in colorsys.hsv_to_rgb((_class / 12) % 1, 1, 1)
        ])
        fig.add_trace(go.Scatter(
            x=df2[df2['class'] == _class]['text_x'],
            y=df2[df2['class'] == _class]['text_y'],
            text=df2[df2['class'] == _class]['name'],
            mode="text",
            textfont=dict(
                # color='#F59626',
                color=_rgb,
                size=18,
                # bgcolor="#ff7f0e",
            ),
            # name=df2[df2['class'] == _class]['name'],
        ))
        for d in df2[df2['class'] == _class].to_dict('records'):
            fig.add_shape(
                type="rect",
                **{
                    k: d[k]
                    for k in ['x0', 'y0', 'x1', 'y1']
                },
                line=dict(
                    width=4,
                    color=_rgb,
                ),
                # fillcolor="LightSkyBlue",
                # name=d['name'],
            )
    fig.update_shapes(dict(xref='x', yref='y'))
    fig.update_traces(showlegend=False)
    
    fig.show(config={'doubleClick': 'reset'})


# %%
# %%
img, d = get_image(76)
print(d['contains_objs'])
img




annotate(
    img,
    df=[
        {
            **{k: v for k, v in zip(['x0', 'y0', 'w', 'h'], v['bbox'])},
            'name': classes_list[(v['category_id'] if v['category_id'] < len(classes_list) else 0)],
            'class': v['category_id'],
        }
        for v in d['anno']
    ]
)





# %%
dist_dict = {
    _id: {
        'id': _id,
        'name': _name,
        'count': 0,
        'image_indices': [],
    }
    for _id, _name in enumerate(classes_list)
}

dist = [
    {
        'id': _id,
        'name': _name,
        'count': np.mean([_id in v['contains_objs'] for v in image_list]),
    }
    for _id, _name in enumerate(classes_list)
]

# %%
for v in image_list:
    for _class in v['contains_objs']:
        dist_dict[_class]['count'] += 1
        # dist_dict[_class]['image_indices'].append(v['image_id'])

# image_list

df = pd.DataFrame(dist_dict.values())
df['ratio'] = df['count'] / len(image_list)
df


# %%
fig = px.line(
    df,
    x='name',
    y=['count'],
    log_y=True,
    width=1600,
    height=1200,
)
fig


# %%