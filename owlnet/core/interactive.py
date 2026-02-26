import torch
from typing import Collection
import ipywidgets as widgets
import numpy as np
import torch.nn.functional as F
from ipywidgets import HBox
from IPython.display import display
import plotly.graph_objects as go
from torchvision import transforms

from owlnet.core.utils import (
    get_label_colours,
    imshow_to_pil,
    get_img_data,
    display_datetime
)
from owlnet.data.dataloading import CollateFunc
from owlnet.core.cluster import get_owlet_clusters


TO_PIL = transforms.ToPILImage()


class VisualiserInteractive:
    def __init__(self, config, embeddings, melspecs, crossing_times, nest_ids) -> None:
        self.embeddings = embeddings
        self.melspecs = melspecs
        self.crossing_times = crossing_times
        self.nest_ids = nest_ids
        self.config = config
        self.owlets = 0

        owlet_clusters, owlet_indices = get_owlet_clusters(self.config, self.embeddings)
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
                batch_sz = self.melspecs[0].shape[0]
                spec_row, spec_col = ind // batch_sz, ind % batch_sz
                spec = self.melspecs[spec_row][spec_col]
                crossing_time = display_datetime(self.crossing_times[ind][0].item())
                nest_id = self.nest_ids[ind]
                imagew.update_layout(
                    images=[
                        dict(source=imshow_to_pil(spec)),
                    ],
                    annotations=[
                        dict(
                            text=f"Time: {crossing_time} | Nest: {nest_id}",
                            x=0.01,
                            y=0.99,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            align="left",
                            font=dict(size=14, color="white"),
                            bgcolor="rgba(0,0,0,0.6)"
                        )
                    ]
                )
            
        self.graph = HBox((figw, imagew))
        for scatterplot in figw.data:
            scatterplot.on_hover(hover_fn)

        min_x, min_y, max_x, max_y= self._get_data_bounds(owlet_clusters)
        self.graph.children[0].update_layout(
            xaxis=dict(range=[min_x, max_x]),  
            yaxis=dict(range=[min_y, max_y]), 
        )

    def add_points(self, points, marker_style, marker_sz):
        figw, _ = self.graph.children
        figw.add_trace(go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='markers',
            marker=dict(size=marker_sz, symbol=marker_style, color="black"),
            name=f"Val pts",
        ))
    
    def pop_verification_trace(self):
        figw, _ = self.graph.children
        num_traces = len(figw.data)
        if num_traces > self.owlets:
            figw.data = figw.data[:num_traces - 1]

    def show(self):
        display(self.graph)
        pass

    def _get_data_bounds(self, embeddings_2d):
        if isinstance(embeddings_2d, Collection):
            embeddings_2d = np.concatenate(embeddings_2d)
        min_x = embeddings_2d[:, 0].min()
        max_x = embeddings_2d[:, 0].max()
        min_y = embeddings_2d[:, 1].min()
        max_y = embeddings_2d[:, 1].max()

        width = max_x - min_x
        height = max_y - min_y

        midpoint_x = (max_x + min_x) / 2
        midpoint_y = (max_y + min_y) / 2

        min_x = (midpoint_x - (width * self.config["axis_correction"] / 2))  
        min_y = (midpoint_y - (height * self.config["axis_correction"] / 2))  
        max_x = (midpoint_x + (width * self.config["axis_correction"] / 2))  
        max_y = (midpoint_y + (height * self.config["axis_correction"] / 2))  
        return min_x, min_y, max_x, max_y 


class ControlPanel:
    def __init__(self, config, embeddings, visualiser):
        self.total_ds_size = embeddings.shape[0]
        self.base_window_width = config["base_window_width"]
        self.hop_sizes = config["hop_sizes"]

        self.hop_size = self.base_window_width
        self.iteration = 0
        self.num_iterations = self.total_ds_size // self.hop_size
        self.collate_func = CollateFunc(spec_height=750)
        self.window_start = 0
        self.display_width = self.total_ds_size // self.base_window_width
        self.window_width = self.hop_size // self.base_window_width

        # Buttons
        self.progress = widgets.Label(value=f"{'█' * self.window_width}{'░'* (self.display_width - self.window_width)}" )
        self.progress_text = widgets.Label(value=f"Dataset slice [0 - {self.hop_size - 1}] of {self.total_ds_size}")
        self.step_button = widgets.Button(description="Step")
        self.reset_button = widgets.Button(description="Reset")
        self.hop_size_buttons = widgets.ToggleButtons(
            options=self.hop_sizes,
            description="Window size (samples) :",
        )
        self.dataset_image = widgets.Image(
            value=get_img_data("img/owlet_full_spectro_large.png"),
            format="png",
            width=1545,
        )

        self.hop_size_buttons.observe(self.on_hop_select, names="value")
        self.step_button.on_click(lambda _: self.step_run(visualiser, embeddings))
        self.reset_button.on_click(lambda _: self.reset(visualiser))

    def on_hop_select(self, change):
        v = int(change["new"])
        if v < self.total_ds_size:
            self.hop_size = v
        else:
            self.hop_size = self.total_ds_size
        
        self.window_width = self.hop_size // self.base_window_width
        self.num_iterations = self.total_ds_size // self.hop_size
        self.iteration = 0
        self.init_progress()

    def init_progress(self):
        self.progress.value  = f"{'█' * self.window_width}{'░'* (self.display_width - self.window_width)}"
        self.progress_text.value = f"Dataset slice [0 - {self.hop_size - 1}] of {self.total_ds_size}"
        
    def step_run(self, visualiser, all_val_embeds):
        if self.iteration > self.num_iterations - 1:
            self.iteration = 0

        bar = ["░"] * self.display_width# Reset bar

        bar_pos = self.iteration * self.window_width
        for i in range(bar_pos,  bar_pos + self.window_width):
            bar[i] = "█"  # Highlight only the window section

        # Update the label
        self.progress.value = "".join(bar)
        self.progress_text.value = f"Dataset slice [{self.iteration * self.hop_size} - {(self.iteration * self.hop_size) + self.hop_size - 1}] of {self.total_ds_size}"
        self.loop_iteration(visualiser, all_val_embeds)


    def reset(self, visualiser):
        self.iteration = 0
        visualiser.pop_verification_trace()
        self.init_progress()


    def loop_iteration(self, visualiser, all_embeds):
        start = self.iteration * self.hop_size
        self.hop_size = min(self.total_ds_size - start, self.hop_size)
        validation_embeds = all_embeds[start: start + self.hop_size]

        visualiser.pop_verification_trace()
        visualiser.add_points(validation_embeds, 'x', 20)
        self.iteration += 1
