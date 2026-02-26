import csv

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from owlnet.data.dataloading import (
    load_data,
    get_verification_dataloader,
    CollateFunc,
    create_embeds
)
from owlnet.core.utils import get_model, reduce_dimensions, get_label_colours
from owlnet.core.cluster import get_owlet_clusters


def export_to_csv(
    filename,
    config,
    sample_rate,
    do_clustering,
    save_plot=False,
    model_name=None):
    """
    Export csv file with the following structure:
    seq_num, timestamp, nest_id, cluster_id, PCA1, PCA2
    """

    owlnet = get_model(config, model_name)
    owlnet = owlnet.eval()
    load_obj = load_data(config, sample_rate)

    owlet_train = load_obj["train"]["dl"]
    embeddings, _, crossing_times, nest_ids = create_embeds(config, owlnet, owlet_train)
    print(f"Created {embeddings.shape[0]} embeddings")

    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings_2d = reduce_dimensions(embeddings)
    print(f"Reduced dimensions")

    nest_lookup = {
        0: "stokewake",
        1: "trigon",
        2: "holtlodge"
    }
    if do_clustering:
        owlet_clusters, owlet_indices = get_owlet_clusters(config, embeddings)
        print(f"Got {len(owlet_clusters)} clusters")

        def find_cluster_of(index):
            for i, indices in enumerate(owlet_indices):
                if index in indices:
                    return i

            assert False, f"Index {index} not found clustering."
            

        rows = []
        for i, item in enumerate(embeddings_2d):
            pc1, pc2 = item.tolist()
            rows.append([
                crossing_times[i][0].item(),
                nest_lookup[nest_ids[i].item()],
                find_cluster_of(i),
                pc1, pc2
            ])


        if save_plot:
            colours = get_label_colours(len(owlet_clusters))
            for i, owlet_cluster in enumerate(owlet_clusters):
                plt.scatter(
                    x=owlet_cluster[:, 0],
                    y=owlet_cluster[:, 1],
                    c=colours[i],
                )
            img_filename = f"{config['exports_dir']}/{filename}.png"
            plt.savefig(img_filename)

        csv_header = ["seq_num", "t_start", "nest_id", "cluster_id", "pc1", "pc2"]
    else: 
        plt.scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            s=1, alpha=0.5, linewidths=0, rasterized=True
        )
        img_filename = f"{config['exports_dir']}/{filename}.png"
        plt.savefig(img_filename)

        rows = []
        for i, item in enumerate(embeddings_2d):
            pc1, pc2 = item.tolist()
            rows.append([
                crossing_times[i][0].item(),
                nest_lookup[nest_ids[i].item()],
                pc1, pc2
            ])
        csv_header = ["seq_num", "t_start", "nest_id", "pc1", "pc2"]


    rows.sort(key=lambda x: x[0])

    with open(f"{config['exports_dir']}/{filename}.csv", "w") as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(csv_header)
        for i, row in enumerate(rows):
            csv_writer.writerow([i] + row)
