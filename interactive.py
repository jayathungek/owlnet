import torch
import ipywidgets as widgets

import torch.nn.functional as F
from utils import get_label_colours, imshow_to_pil
from data import get_verification_dataloader, CollateFunc
from cluster import get_owlet_clusters
from ipywidgets import HBox
from IPython.display import display
import plotly.graph_objects as go
from torchvision import transforms


TO_PIL = transforms.ToPILImage()


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


def create_embeds(model, dataloader):
    model.eval()
    embeds = []
    specs = []
    specs_og = []
    for batch in dataloader:
        with torch.no_grad():
            data_specs, og_specs = batch
            specs_og += og_specs.unbind()
            specs += data_specs.unbind()
            data_specs = data_specs.cuda()
            embeds_batch = model(data_specs.cuda())
            embeds.append(embeds_batch.detach().cpu())
    embeds = torch.cat(embeds)
    return embeds, specs, specs_og

def get_img_data(img_path):
    with open(img_path, "rb") as fh:
        data = fh.read()
    return data

# Control flag
total_ds_size = 3375
base_window_width = 20

HOP_SIZES = [20, 40, 60, 80, 100, 200, 300, 500]
hop_size = base_window_width
iteration = 0
num_iterations = total_ds_size // hop_size
collate_func = CollateFunc(spec_height=750)

window_start = 0
display_width = total_ds_size // base_window_width
window_width = hop_size // base_window_width

# Buttons
progress = widgets.Label(value=f"{'█' * window_width}{'░'* (display_width - window_width)}" )
progress_text = widgets.Label(value=f"Dataset slice [0 - {hop_size - 1}] of {total_ds_size}")
step_button = widgets.Button(description="Step")
reset_button = widgets.Button(description="Reset")
hop_size_buttons = widgets.ToggleButtons(
    options=HOP_SIZES,
    description="Hop size:",
)


dataset_image = widgets.Image(
    value=get_img_data("img/owlet_full_spectro_large.png"),
    format="png",
    width=1545,
)


def on_hop_select(change):
    global iteration
    global num_iterations
    global total_ds_size
    global hop_size 
    global progress
    global window_width

    v = int(change["new"])
    if v < total_ds_size:
        hop_size = v
    else:
        hop_size = total_ds_size
    
    window_width = hop_size // base_window_width
    num_iterations = total_ds_size // hop_size
    iteration = 0
    init_progress()

hop_size_buttons.observe(on_hop_select, names="value")

def init_progress():
    global progress
    global progress_text
    global hop_size
    global total_ds_size
    global window_width
    global display_width

    progress.value  = f"{'█' * window_width}{'░'* (display_width - window_width)}"
    progress_text.value = f"Dataset slice [0 - {hop_size - 1}] of {total_ds_size}"
    

def on_text_submit(change):
    global iteration
    global num_iterations
    global total_ds_size
    global hop_size 
    global progress
    v = int(change.value)
    if v < total_ds_size:
        hop_size = v
    else:
        hop_size = total_ds_size
    
    num_iterations = total_ds_size // hop_size
    iteration = 0
    init_progress()


def step_run(dataset, model, visualiser):
    global iteration
    global num_iterations
    global progress
    global progress_text
    global window_width
    global display_width

    if iteration > num_iterations - 1:
        iteration = 0

    bar = ["░"] * display_width# Reset bar

    bar_pos = iteration * window_width
    for i in range(bar_pos,  bar_pos + window_width):
        bar[i] = "█"  # Highlight only the window section

    # Update the label
    progress.value = "".join(bar)
    progress_text.value = f"Dataset slice [{iteration * hop_size} - {(iteration * hop_size) + hop_size - 1}] of {total_ds_size}"
    loop_iteration(dataset, model, visualiser)


def reset(visualiser):
    global iteration
    iteration = 0
    visualiser.pop_verification_trace()
    init_progress()


def loop_iteration(owlet_dataset, owlnet, visualiser):
    global iteration
    global hop_size
    global total_ds_size
    global num_iterations
    global collate_func

    start = iteration * hop_size
    hop_size = min(total_ds_size - start, hop_size)
    indices = [start, start + hop_size]
    verification_dl = get_verification_dataloader(owlet_dataset, indices, collate_func)
    validation_embeds, _, _ = create_embeds(owlnet, verification_dl)
    validation_embeds = F.normalize(validation_embeds, p=2, dim=1)

    visualiser.pop_verification_trace()
    visualiser.add_points(validation_embeds, 'x', 20)
    iteration += 1
