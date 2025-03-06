import torch
import math
import ipywidgets as widgets

import torch.nn.functional as F
from data import get_verification_dataloader, CollateFunc


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

HOP_SIZES = [20, 30, 40, 50, 100, 200, 500]
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
# hop_size_input = widgets.Text(
#     description="Hop size:",
#     placeholder=f"{hop_size}"
# )
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

# hop_size_input.on_submit(on_text_submit)

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


# for start in range(0, total_ds_size, hop_size):
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


