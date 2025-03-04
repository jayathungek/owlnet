import torch
import time
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

import torch.nn.functional as F
from data import get_verification_dataloader, CollateFunc


def create_embeds(model, dataloader):
    embeds = []
    specs = []
    specs_og = []
    for batch in dataloader:
        data_specs, og_specs = batch
        specs_og += og_specs.unbind()
        specs += data_specs.unbind()
        data_specs = data_specs.cuda()
        embeds_batch = model(data_specs.cuda())
        embeds.append(embeds_batch.detach().cpu())
    embeds = torch.cat(embeds)
    return embeds, specs, specs_og

# Control flag
auto_run = False
total_ds_size = 3375
hop_size = 50
iteration = 0
num_iterations = total_ds_size // hop_size
collate_func = CollateFunc(spec_height=750)
window_start = 0

# Buttons
progress = widgets.Label(value=f"█{'░'* (num_iterations - 1)}" )
step_button = widgets.Button(description="Step")
reset_button = widgets.Button(description="Reset")


def step_run(dataset, model, visualiser):
    global auto_run
    global iteration
    global num_iterations

    if iteration > num_iterations - 1:
        iteration = 0

    auto_run = False
    bar = ["░"] * num_iterations# Reset bar
    bar[iteration] = "█"  # Highlight only the window section

    # Update the label
    progress.value = "".join(bar)
    loop_iteration(dataset, model, visualiser)


def reset(visualiser, progress_widget):
    global iteration
    global auto_run
    iteration = 0
    progress_widget.value = f"█{'░'* (num_iterations - 1)}" 
    auto_run = False
    visualiser.pop_verification_trace()


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
    visualiser.pop_verification_trace()
    validation_embeds = F.normalize(validation_embeds, p=2, dim=1)
    visualiser.add_points(validation_embeds, 'x', 20)
    iteration += 1


