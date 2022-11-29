import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _mesh_graph_object(pos, faces, intensity=None, scene="scene", showscale=True):
    cpu = torch.device("cpu")
    if type(pos) != np.ndarray:
        pos = pos.to(cpu).clone().detach().numpy()
    if pos.shape[-1] != 3:
        raise ValueError("Vertices positions must have shape [n,3]")
    if type(faces) != np.ndarray:
        faces = faces.to(cpu).clone().detach().numpy()
    if faces.shape[-1] != 3:
        raise ValueError("Face indices must have shape [m,3]")
    if intensity is None:
        intensity = np.ones([pos.shape[0]])
    elif type(intensity) != np.ndarray:
        intensity = intensity.to(cpu).clone().detach().numpy()

    x, z, y = pos.T
    i, j, k = faces.T

    mesh = go.Mesh3d(x=x, y=y, z=z,
                     color='lightpink',
                     intensity=intensity,
                     opacity=1,
                     colorscale=[[0, 'gold'], [0.5, 'mediumturquoise'], [1, 'magenta']],
                     i=i, j=j, k=k,
                     showscale=showscale,
                     scene=scene)
    return mesh


def visualize_and_compare(adv_pos, faces, orig_pos, orig_faces, intensity=None):
    orig_mesh = _mesh_graph_object(orig_pos, orig_faces, intensity, "scene")
    adv_mesh = _mesh_graph_object(adv_pos, faces, intensity, "scene2")
    """
    # Superimpose original shape to compare.
    n, m = original_pos.shape[0], pos.shape[0]
    compare_pos = torch.cat([original_pos, pos], dim=0)
    compare_faces = torch.cat([original_faces, faces + n], dim=0)
    compare_color = torch.zeros([n + m], dtype=pos.dtype, device=pos.device)
    compare_color[n:] = (pos - original_pos).norm(p=2, dim=-1)

    mesh_cmp = _mesh_graph_object(compare_pos, compare_faces, compare_color, "scene2", showscale=False)

    """
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "scene"}, {"type": "scene"}]])

    fig.add_trace(
        orig_mesh,
        row=1, col=1
    )

    fig.add_trace(
        adv_mesh,
        row=1, col=2
    )

    fig.update_layout(
        scene=go.layout.Scene(aspectmode="data"),
        scene2=go.layout.Scene(aspectmode="data"),
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue"
    )

    fig.show()
    return


def visualize(pos, faces, intensity=None):
    mesh = _mesh_graph_object(pos, faces, intensity)
    layout = go.Layout(scene=go.layout.Scene(aspectmode="data"))

    # pio.renderers.default="plotly_mimetype"
    fig = go.Figure(data=[mesh],
                    layout=layout)
    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue")
    fig.show()
    return


def compare(pos1, faces1, pos2, faces2):
    n, m = pos1.shape[0], pos2.shape[0]
    tmpx = torch.cat([pos1, pos2], dim=0)
    tmpf = torch.cat([faces1, faces2 + n], dim=0)
    color = torch.zeros([n + m], dtype=pos1.dtype, device=pos1.device)
    color[n:] = (pos1 - pos2).norm(p=2, dim=-1)
    visualize(tmpx, tmpf, color)
