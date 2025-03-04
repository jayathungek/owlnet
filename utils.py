import io
import colorsys
import torch
import numpy as np
from typing import Collection
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
from umap import UMAP
import numpy as np
from ipywidgets import HBox
from IPython.display import display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
from sklearn.decomposition import PCA
from cluster import get_owlet_clusters


TO_PIL = transforms.ToPILImage()


def imshow_to_pil(image_array, cmap="viridis"):
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.axis("off")  # Remove axes
    image_array = image_array.squeeze()
    ax.imshow(image_array, cmap=cmap, aspect="auto", origin="lower")

    # Save figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Convert buffer to PIL Image
    buf.seek(0)
    pil_image = Image.open(buf)
    
    return pil_image

def reduce_dimensions(embeddings):
    # reducer =UMAP(n_components=2, metric="cosine")
    # reducer.fit(embeddings)
    # tsne_points = reducer.transform(embeddings)

    
    reducer = PCA(n_components=2, svd_solver='auto')
    tsne_points = reducer.fit_transform(embeddings)
    return tsne_points


def get_label_colours(n):
    colors = []
    hue_step = 360.0 / n

    for i in range(n):
        hue = i * hue_step
        saturation = 1  # You can adjust saturation and lightness if needed
        lightness = 0.4   # You can adjust saturation and lightness if needed

        rgb = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)
        hexcol = "#" + "".join([f"{int(v * 255):02X}" for v in rgb])
        colors.append(hexcol)

    return colors
    


class VisualiserInteractive:
    def __init__(self, embeddings, melspecs, melspecs_og) -> None:
        self.embeddings = embeddings
        self.melspecs = melspecs
        self.melspecs_og = melspecs_og
        self.owlets = 0

        owlet_clusters, owlet_indices = get_owlet_clusters(self.embeddings)
        colours = get_label_colours(len(owlet_clusters))

        
        figw = go.FigureWidget()
        
        for i, owlet_cluster in enumerate(owlet_clusters):
            new_customdata = owlet_indices[i].tolist()
            print(f"Adding {len(new_customdata)} points for Owlet {i + 1}")
            figw.add_trace(go.Scatter(
                customdata=new_customdata,
                x=owlet_cluster[:, 0],
                y=owlet_cluster[:, 1],
                mode='markers',
                marker=dict(size=3, color=colours[i]),
                name=f"Owlet {i + 1}",
            ))
            self.owlets += 1
            


        figw.update_layout(
            title="Hover over points to view spectrogram",
            hovermode="closest",
            xaxis=dict(title="Component 1", scaleanchor="y"),  # Lock x-axis to y-axis scale
            yaxis=dict(title="Component 2"),
            width=600,  # Set fixed width
            height=600,  # Set fixed height
        )

        image = go.Figure()
        image.add_layout_image(
            dict(
                source=TO_PIL(torch.zeros(3, 128, 400)),
                xref="x",
                yref="y",
                x=0,
                y=3,
                sizex=2,
                sizey=2,
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        )

        image.update_layout(
            xaxis=dict(
                showgrid=False,  # Hide grid lines
                zeroline=False,  # Hide zero line
                showticklabels=False,  # Hide tick labels
            ),
            yaxis=dict(
                showgrid=False,  # Hide grid lines
                zeroline=False,  # Hide zero line
                showticklabels=False,  # Hide tick labels
            ),
            plot_bgcolor="white",  # Set the background to white (optional)
            margin=dict(t=0, b=0, l=0, r=0),  # Remove any extra margins
            xaxis_visible=False,  # Hide the x-axis
            yaxis_visible=False,  # Hide the y-axis
        )
        imagew = go.FigureWidget(image)

        def hover_fn(trace, point, selector):
            if len(point.point_inds) > 0:
                ind = point.point_inds[0]
                spec = self.melspecs[ind]
                imagew.update_layout(
                    images=[
                        dict(source=imshow_to_pil(spec)),
                    ]
                )


            
            
        self.graph = HBox((figw, imagew))
        for scatterplot in figw.data:
            scatterplot.on_hover(hover_fn)

    def add_points(self, points, marker_style, marker_sz):
        figw, _ = self.graph.children
        figw.add_trace(go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='markers',
            marker=dict(size=marker_sz, symbol=marker_style, color="black"),
            name=f"Val data",
        ))
    
    def pop_verification_trace(self):
        figw, _ = self.graph.children
        num_traces = len(figw.data)
        if num_traces > self.owlets:
            figw.data = figw.data[:num_traces - 1]

    def show(self):
        display(self.graph)
        pass

    

# import time
# import torch.nn.functional as F

# total_ds_size = 3375
# hop_size = 20
# for start in range(0, total_ds_size, hop_size):
#     hop_size = min(total_ds_size - start, hop_size)
#     indices = [start, start + hop_size]
#     verification_dl = get_verification_dataloader(owlet_dataset, indices, collate_func)
#     validation_embeds, _, _ = create_embeds(owlnet, model_name, verification_dl)
#     vis.pop_verification_trace()
#     validation_embeds = F.normalize(validation_embeds, p=2, dim=1)
#     vis.add_points(validation_embeds, 'x', 5)
#     time.sleep(1.0)
    
