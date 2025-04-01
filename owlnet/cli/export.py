import csv

import torch.nn.functional as F
import matplotlib.pyplot as plt

from owlnet.data.dataloading import (
    load_data,
    get_verification_dataloader,
    CollateFunc,
    create_embeds
)
from owlnet.core.utils import get_model, reduce_dimensions


def export_to_csv(filename, config, save_plot=False, model_name=None):
    """
    Export csv file with the following structure:
    seq_num, t_start, t_end, PCA1, PCA2
    """
    owlnet = get_model(config, model_name)
    owlnet = owlnet.eval()
    _, _, owlet_ds = load_data(config)

    collate_func = CollateFunc(config["spec_height"])
    dataloader = get_verification_dataloader(owlet_ds, None, collate_func)
    embeddings, _, _, crossing_times = create_embeds(config, owlnet, dataloader)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings_2d = reduce_dimensions(embeddings)

    rows = []
    for i, item in enumerate(embeddings_2d):
        pc1, pc2 = item.tolist()
        rows.append([i, crossing_times[i][0], crossing_times[i][1], pc1, pc2])

    with open(f"{config['exports_dir']}/{filename}.csv", "w") as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(["seq_num", "t_start", "t_end", "pc1", "pc2"])
        csv_writer.writerows(rows)

    if save_plot:
        img_filename = f"{config['exports_dir']}/{filename}.png"
        plt.scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], s=10)
        plt.savefig(img_filename)

    return embeddings_2d, crossing_times
