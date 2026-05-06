import re
import json
from tqdm import tqdm
from pathlib import Path

import torch
import webdataset as wds

try:
    from owlnet.core.utils import (
        infer_abs_unix_timestamp,
        get_nest_id,
        chop_file
    )
except ModuleNotFoundError:
    from ..core.utils import (
        infer_abs_unix_timestamp,
        get_nest_id,
        chop_file
    )
 

def split_into_nests(wavs):
    nests = {}
    for file, start_time, nestid in wavs:
        if nestid not in nests.keys():
            nests[nestid] = []
        nests[nestid].append((file, start_time))
    return nests


def extract_features(config):
    ignore = config["ignore"]
    all_wavs = []
    for d in list(Path(config["data_dir"]).glob("**/*.[Ww][Aa][Vv]")):
        if ignore is None:
            to_keep = True
        else:
            to_keep = re.search(ignore, d.stem) is None
        if to_keep:
            all_wavs.append((d, infer_abs_unix_timestamp(d.stem), get_nest_id(d)))

    all_wavs = sorted(
        all_wavs,
        key=lambda x:  x[1]
    )

    nests = split_into_nests(all_wavs)
    global_index = 0
    failed_filenames = []
    ds_info = {"nests": {}}
    for nestid, samples in nests.items():
        shard_pattern = config["shard_pattern"].format(
            sharddir=config["shard_dir"], 
            nestid=nestid
        )
        shard_size = config["shard_size"]
        with wds.ShardWriter(shard_pattern, maxcount=shard_size) as sink:
            shard_index = 0
            for file, start_time in tqdm(samples):
                threshold = config["zero_threshold"]
                print(f"processing file {file.stem} with threshold: {threshold}")
                try:
                    chunk, chunks_crossing_times = chop_file(
                        config,
                        file,
                        start_time
                    )
                except RuntimeError:
                    print(f"Failed to open file {file.stem}")
                    failed_filenames.append(file.stem)
                    continue
                for chunk, timestamps in zip(chunk, chunks_crossing_times):
                    sample = {
                        "__key__": f"{shard_index:09d}",
                        "tensor.pth": chunk,
                        "timestamp.pth": torch.tensor(timestamps, dtype=torch.float64),
                        "nestid.pth": torch.tensor(nestid, dtype=torch.int64)
                    }
                    sink.write(sample)
                    shard_index += 1
                    global_index += 1
                print(f"{global_index} data points so far...")
            ds_info["nests"][nestid] = {"num_samples": shard_index}
            ds_info["nests"][nestid]["num_shards"] = sink.shard

        ds_info["failed_filenames"] = failed_filenames
        print(f"Got {global_index} data points")
    with open(f"{config['shard_dir']}/ds_info.json", "w") as fh:
        json.dump(ds_info, fh)