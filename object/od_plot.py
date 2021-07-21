# %%
import enum
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import os, time
import torch
from PIL import Image
import colorsys

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
            for v in colorsys.hsv_to_rgb((_class / 80) % 1, 1, 1)
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