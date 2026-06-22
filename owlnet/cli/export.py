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
    embeddings, _, crossing_times, nest_ids, dreiss_features = create_embeds(config, owlnet, owlet_train)
    print(f"Created {embeddings.shape[0]} embeddings, {dreiss_features.shape} Dreiss features")

    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings_2d = reduce_dimensions(embeddings)
    print(f"Reduced dimensions")

    nest_lookup = {
        0: "stokewake",
        1: "trigon",
        2: "holtlodge",
        6: "testnest",
        7: "testnest",
    }
    sr_lookup = {
        0: 48_000,
        1: 24_000,
        2: 24_000,
        6: 24_000,
        7: 24_000,
    }
    SPEC_HEIGHT = 2049
    N_FFT = config["n_fft"]
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
            sr = sr_lookup[nest_ids[i].item()]
            nyquist_freq = sr / 2
            pc1, pc2 = item.tolist()
            rows.append([
                crossing_times[i][0].item(),
                crossing_times[i][1].item(),
                nest_lookup[nest_ids[i].item()],
                find_cluster_of(i),
                dreiss_features[i][1].item(),
                dreiss_features[i][2].item() * nyquist_freq,
                dreiss_features[i][3].item() * nyquist_freq,
                dreiss_features[i][4].item() * nyquist_freq,
                pc1, pc2
            ])


        if save_plot:
            colours = get_label_colours(len(owlet_clusters))
            plt.figure(figsize=(10,10))
            for i, owlet_cluster in enumerate(owlet_clusters):
                plt.scatter(
                    x=owlet_cluster[:, 0],
                    y=owlet_cluster[:, 1],
                    c=colours[i],
                    s=1, alpha=0.5, linewidths=0, rasterized=True
                )
            img_filename = f"{config['exports_dir']}/{filename}.png"
            plt.savefig(img_filename, dpi=300, bbox_inches="tight")

        csv_header = ["seq_num", "t_start", "t_end", "nest_id", "cluster_id", "loudness_deviation", "mean_freq", "upper_freq", "freq_variation", "pc1", "pc2"]
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
            sr = sr_lookup[nest_ids[i].item()]
            nyquist_freq = sr / 2
            pc1, pc2 = item.tolist()
            rows.append([
                crossing_times[i][0].item(),
                crossing_times[i][1].item(),
                nest_lookup[nest_ids[i].item()],
                dreiss_features[i][1].item(),
                dreiss_features[i][2].item() * nyquist_freq,
                dreiss_features[i][3].item() * nyquist_freq,
                dreiss_features[i][4].item() * nyquist_freq,
                pc1, pc2
            ])
        csv_header = ["seq_num", "t_start", "t_end", "nest_id", "loudness_deviation", "mean_freq", "upper_freq", "freq_variation","pc1", "pc2"]

    rows.sort(key=lambda x: x[0])

    with open(f"{config['exports_dir']}/{filename}.csv", "w") as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(csv_header)
        for i, row in enumerate(rows):
            csv_writer.writerow([i] + row)
