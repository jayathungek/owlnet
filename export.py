import sys
import csv
from argparse import ArgumentParser
import torch.nn.functional as F

import matplotlib.pyplot as plt
from data import load_data, get_verification_dataloader, CollateFunc
from utils import load_config, get_model, reduce_dimensions
from interactive import create_embeds


def export_to_csv(model_name, filename, config, save_plot=False):
    """
    Export csv file with the following structure:
    seq_num, t_start, t_end, PCA1, PCA2
    """
    owlnet = get_model(config, model_name, attention=False)
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

    with open(f"{config['exports_dir']}/{filename}", "w") as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(["seq_num", "t_start", "t_end", "pc1", "pc2"])
        csv_writer.writerows(rows)

    if save_plot:
        img_filename = f"{config['exports_dir']}/{parsed_args.filename.split('.')[0]}.png"
        plt.scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], s=10)
        plt.savefig(img_filename)

    return embeddings_2d, crossing_times


def parse_args(args):
    parser = ArgumentParser(prog="export")
    parser.add_argument("filename")
    parser.add_argument("-c", "--config", default="config.json")
    parser.add_argument("-m", "--model", default="model.v1_3584.datapoints_105.epochs")
    parsed_args = parser.parse_args(args)
    return parsed_args
    

if __name__ == "__main__":
    parsed_args = parse_args(sys.argv[1:])
    config = load_config(parsed_args.config)
    embeds_2d, crossing_times = export_to_csv(parsed_args.model, parsed_args.filename, config, save_plot=True)
    