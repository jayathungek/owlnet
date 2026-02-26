import sys
from argparse import ArgumentParser, BooleanOptionalAction

from owlnet.core.utils import load_config


def export_parser(args):
    parser = ArgumentParser(prog="export")
    parser.add_argument("filename")
    parser.add_argument("sample_rate", type=float)
    parser.add_argument("-k", "--clustering", action=BooleanOptionalAction, default=False)
    parser.add_argument("-c", "--config", default="settings/config.json")
    parser.add_argument("-m", "--model")
    parsed_args = parser.parse_args(args)
    return parsed_args


def train_parser(args):
    parser = ArgumentParser(prog="train")
    parser.add_argument("model_name")
    parser.add_argument("-c", "--config", default="settings/config.json")
    parsed_args = parser.parse_args(args)
    return parsed_args


def sharding_parser(args):
    parser = ArgumentParser(prog="shard")
    parser.add_argument("config")
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
        from owlnet.core.train import train
        train_args = train_parser(sub_args)
        config = load_config(train_args.config)
        train(config, train_args.model_name)
    elif parsed.program == "export":
        from owlnet.cli.export import export_to_csv
        export_args = export_parser(sub_args)
        config = load_config(export_args.config)
        export_to_csv(
            export_args.filename,
            config,
            export_args.sample_rate,
            export_args.clustering,
            save_plot=True,
            model_name=export_args.model
        )
    elif parsed.program == "shard":
        from owlnet.data.sharding import extract_features
        shard_args = sharding_parser(sub_args)
        config = load_config(shard_args.config)
        extract_features(config)
    else:
        raise ValueError(f"Program {parsed.program} unknown, please use one of: train, export")

