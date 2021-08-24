# %%
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import os, time
import torch
from PIL import Image
import colorsys
import json, string, re

os.environ['TORCH_HOME'] = '/host/ubuntu/torch'

# %%
dp = '/home/researcher/VT/ViT_torch/data/od'
dp = '/host/ubuntu/torch/coco2017/validation/data'
fps = [os.path.join(dp, v) for v in os.listdir(dp)]
fps[:5], len(fps)


# %%
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# %%
# Images
imgs = [*fps]

# Inference
results = model(imgs)

# Results
results.print()
# results.save()  # or .show()

# results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]

# %%
# def draw():

# %%
def annotate(fp, df=None, model=None, max_size=1200):
    # fp = fps[0]
    if isinstance(fp, Image.Image):
        img = fp
    elif isinstance(fp, (list)):
        imgs = [Image.open(_fp) for _fp in fp]
        feeds = [*imgs]
        results = model(feeds)
        return [
            annotate(_img, _df, None, max_size=max_size)
            for _img, _df in zip(imgs, results.pandas().xyxy)
        ]
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
    
    df2 = df.rename(columns={
        'xmin': 'x0',
        'xmax': 'x1',
        'ymin': 'y0',
        'ymax': 'y1',
    }).copy(True)
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
annotate(
    fp=fps[209],
    # df=results.pandas().xyxy[1],
    model=model,
    max_size=1200,
)
# %%


# %%
label_fp = '/host/ubuntu/torch/coco2017/validation/labels.json'
label_json = json.load(open(label_fp, 'r'))
len(label_json)

# %%
annos = label_json['annotations']
imgs = label_json['images']

# %%
labels = {}
for _anno in annos:
    # ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
    _image_id = _anno['image_id']
    if _image_id not in labels:
        labels[_image_id] = {
            'image_id': _image_id,
            'bboxes': [],
            'file_name': None,
            'file_path': None,
            'found_image': False,
            'height': None,
            'width': None,
        }
    labels[_image_id]['bboxes'].append({
        'id': _anno['id'],
        'class': _anno['category_id'],
        'bbox': _anno['bbox'],
        'iscrowd': _anno['iscrowd'],
    })

_miss_count = 0
for _img in imgs:
    # ['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id']
    _image_id = _img['id']
    if _image_id not in labels:
        _miss_count += 1
        continue
    labels[_image_id]['height'] = _img['height']
    labels[_image_id]['width'] = _img['width']
    labels[_image_id]['file_name'] = _img['file_name']
    labels[_image_id]['found_image'] = True

labels = {
    k: v
    for k, v in labels.items()
    if v['found_image']
}

print('annos[{}] imgs[{}] matched[{}]'.format(
    len(annos), len(imgs), len(labels)
))

# %%
labels_list = list(labels.values())
len(labels_list)

# %%
labels_list[0]

# %%
all_labels_train = [
            # 'background',
            'airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench',
            'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot',
            'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut',
            'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse',
            'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange',
            'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich',
            'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign',
            'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush',
            'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']

# %%
for i in np.random.randint(4000, size=1):
    annotate(
        fp=os.path.join(dp, labels_list[i]['file_name']),
        # df=results.pandas().xyxy[1],
        df=pd.DataFrame([
            {
                # **{
                #     k: v
                #     for k, v in zip(['ymin', 'ymax', 'xmin', 'xmax'], bb['bbox'])
                # },
                'xmin': bb['bbox'][0],
                'xmax': bb['bbox'][1] + bb['bbox'][0],
                'ymin': bb['bbox'][2],
                'ymax': bb['bbox'][3] + bb['bbox'][2],
                'class': bb['class'],
                'name': all_labels_train[min(bb['class'], 79)],
            }
            for bb in labels_list[i]['bboxes']
        ]),
        # model=model,
        max_size=1200,
    )

# %%
annotate(
    fp=[os.path.join(dp, labels_list[i]['file_name'])
        for i in np.random.randint(4000, size=5)
    ],
    # df=results.pandas().xyxy[1],
    model=model,
    max_size=2000,
)


# %%

# %% plot raw targets
import json, os, plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# %%
fp = '/home/researcher/VT/ViT_torch/object/logs/temp_transform.json'
fp = '/home/researcher/VT/ViT_torch/object/logs/temp_transform2.json'
a = json.load(open(fp, 'r'))
len(a)

# %%
data = []
miss_count = 0
for i, v in enumerate(zip(a['before'], a['after'])):
    # data.append({
    #     'index': i,
    #     **{
    #         '{}{}'.format(k[:1], j): v[j][k]
    #         for k in ['bbox', 'segmentation', 'area']
    #         for j in range(2)
    #     },
    # })
    for j in range(2):
        for seg in v[j]['segmentation']:
            for j1 in range(len(seg) // 2):
                data.append({
                    'index': i,
                    'state': j,
                    'type': 's' + str(j),
                    **{k: seg[j1 * 2 + j2]
                        for j2, k in enumerate(list('xy'))},
                })
        
        
        # data.append({
        #     'index': i,
        #     'state': j,
        #     'type': None,
        #     'x': None,
        #     'y': None,
        # })
        b = v[j]['bbox']
        # b2 = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
        if b[2] < b[0] or b[3] < b[1]:
            miss_count += 1
        for j1, x in enumerate([b[0], b[0] + b[2]]):
            for y in [b[1], b[1] + b[3]][::(1 if j1 == 0 else -1)]:
                data.append({
                    'index': i,
                    'state': j,
                    'type': 'b' + str(j),
                    'x': x,
                    'y': y,
                })
        
        # data.append({
        #     'index': i,
        #     'state': j,
        #     'type': None,
        #     'x': None,
        #     'y': None,
        # })

len(data)
print('miss_count:', miss_count)

df = pd.DataFrame(data)
df

# %%
px.line(
    df[df['index']==2],
    x='x',
    y='y',
    color='type',
    hover_data=['index'],
)

# %%
