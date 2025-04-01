import sys
from argparse import ArgumentParser

from owlnet.core.train import train
from owlnet.cli.export import export_to_csv
from owlnet.core.utils import load_config


def export_parser(args):
    parser = ArgumentParser(prog="export")
    parser.add_argument("filename")
    parser.add_argument("-c", "--config", default="settings/config.json")
    parsed_args = parser.parse_args(args)
    return parsed_args


def train_parser(args):
    parser = ArgumentParser(prog="train")
    parser.add_argument("model_name")
    parser.add_argument("-c", "--config", default="settings/config.json")
    parsed_args = parser.parse_args(args)
    return parsed_args


def parse_args(args):
    parser = ArgumentParser(prog="owlnet.cli")
    parser.add_argument("program")
    parsed_args = parser.parse_args(args)
    return parsed_args

    
if __name__ == "__main__":
    args = sys.argv[1:]
    sub_args = args[1:]
    parsed = parse_args([args[0]])
    if parsed.program == "train":
        train_args = train_parser(sub_args)
        config = load_config(train_args.config)
        train(config, train_args.model_name)
    elif parsed.program == "export":
        export_args = export_parser(sub_args)
        config = load_config(export_args.config)
        embeds_2d, crossing_times = export_to_csv(
            config["default_model"],
            export_args.filename,
            config,
            save_plot=True
        )
    else:
        raise ValueError(f"Program {parsed.program} unknown, please use one of: train, export")

