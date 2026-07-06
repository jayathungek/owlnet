from typing import Union
import glob

import webdataset as wds

from owlnet.core.utils import *
from owlnet.data.augment import Specaugment
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import resize


def get_verification_dataloader(full_dataset, verification_subset, collate_fn):
    if verification_subset is not None:
        start, end = verification_subset
    else:
        start = 0
        end = len(full_dataset)

    indices = list(range(start, end))
    verification_dataset = Subset(full_dataset, indices)
    dataloader = DataLoader(
        dataset=verification_dataset, 
        collate_fn=collate_fn,
        batch_size=len(indices), 
        shuffle=False
    )            
    return dataloader


class CollateFunc:

    def __init__(self, config):
        self.spec_height = config['spec_height']
        self.min_len = config['min_tsteps']
        self.augmentor = Specaugment(
            config['specaugment_tmask'],
            config['specaugment_fmask']
        )
    
    def __call__(self, batch):
        max_len = self.min_len
        for spec, _, _, _ in batch:
            time_len = spec.shape[-1]
            max_len = time_len if time_len > max_len else max_len
            
        
        padded = []
        crossing_times_list = []
        nest_id_list = []
        dreiss_features_list = []
        for spec, dreiss_features, crossing_times, nest_id in batch:
            padding_needed = max_len - spec.shape[2]
            if padding_needed > 0:
                height = spec.shape[1]
                spec = resize(spec, (height, max_len), antialias=True)
                # spec = F.pad(spec, (0, padding_needed), "constant", 0)

            padded.append(spec)
            nest_id_list.append(nest_id.unsqueeze(0))
            crossing_times_list.append(crossing_times.unsqueeze(0))
            dreiss_features_list.append(dreiss_features.unsqueeze(0))
            
        padded = torch.cat(padded).unsqueeze(1)
        crossing_times = torch.cat(crossing_times_list)
        nest_ids = torch.cat(nest_id_list)
        dreiss_features = torch.cat(dreiss_features_list)
        padded_aug = self.augmentor(padded)
        return padded, padded_aug, dreiss_features, crossing_times, nest_ids
    

def load_data(config, sample_rate=1.0):
    # Union[str, list]="all" # if not "all", list all nests to include in train dl, rest go to test/other dl
    nests_split = config["split"]
    with open(f"{config['proj_root']}/{config['shard_dir']}/ds_info.json", "r") as fh:
        ds_info =  json.load(fh)
    ds_nests = set([int(k) for k in ds_info["nests"].keys()])

    if isinstance(nests_split, str):
        if nests_split == "all":
            train_nests = list(ds_nests)
            test_nests = set([])
        else:
            assert False, f"Invalid str {nests_split} for nests_split. Must be `all`."
    else:
        train_nests = set(nests_split[0])
        test_nests = set(nests_split[1])
        assert  train_nests.isdisjoint(test_nests), f"Nest split must be disjoint!"
        assert len(train_nests) > 0, f"No. of train nests must be > 0"
        user_nests = train_nests | test_nests
        assert user_nests <= ds_nests, f"Nest split contains nest IDs that are not in dataset: {list(user_nests - ds_nests)}"
        

    collate_func = CollateFunc(config)

    train_shards = []
    for nestid in train_nests:
        train_shards += sorted(glob.glob(f"{config['proj_root']}/{config['shard_dir']}/nest-{nestid}-*.tar"))
    train_ds = (
        wds.WebDataset(
            train_shards,
            shardshuffle=config['shard_size']
        )
           .shuffle(config['shard_size'])
           .rsample(sample_rate)
           .decode("torch")
           .to_tuple("tensor.pth", "dreiss_features.pth", "timestamp.pth", "nestid.pth")
    )
    train_dl = DataLoader(
        train_ds, 
        collate_fn=collate_func,
        num_workers=config['num_ds_workers'],
        batch_size=config['batch_sz']
    )            
    train_size = 0
    for nestid in train_nests:
        train_size += ds_info["nests"][str(nestid)]["num_samples"]

    if len(test_nests) > 0:
        test_shards = []
        for nestid in test_nests:
            test_shards += sorted(glob.glob(f"{config['proj_root']}/{config['shard_dir']}/nest-{nestid}-*.tar"))
        test_ds = (
            wds.WebDataset(
                test_shards,
                shardshuffle=config['shard_size']
            )
            .shuffle(config['shard_size'])
            .decode("torch")
            .to_tuple("tensor.pth", "dreiss_features.pth", "timestamp.pth", "nestid.pth")
        )
        test_dl = DataLoader(
            test_ds, 
            collate_fn=collate_func,
            num_workers=config['num_ds_workers'],
            batch_size=config['batch_sz']
        )            
        test_size = 0
        for nestid in test_nests:
            test_size += ds_info["nests"][str(nestid)]["num_samples"]
    else:
        test_ds = None
        test_dl = None
        test_size = 0

    load_results = {
        "train": {
            "dl": train_dl,
            "ds": train_ds,
            "size": train_size
        },
        "test": {
            "dl": test_dl,
            "ds": test_ds,
            "size": test_size
        },
    }

    return load_results

def create_embeds(config, model, dataloader):
    model.eval()
    embeds = []
    specs = []
    crossing_times_list = []
    nest_id_list = []
    dreiss_features_list = []
    for batch in dataloader:
        with torch.no_grad():
            data_specs, _, dreiss_features, crossing_times, nest_ids = batch
            specs.append(data_specs)
            data_specs = data_specs.to(config['device'])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True), torch.backends.cuda.sdp_kernel(enable_flash=False):
                embeds_batch = model(data_specs, dreiss_features)
                embeds_batch = F.normalize(embeds_batch, p=2, dim=1) 
                embeds.append(embeds_batch.detach().cpu())
                nest_id_list.append(nest_ids)
                crossing_times_list.append(crossing_times)
                dreiss_features_list.append(dreiss_features)
    embeds = torch.cat(embeds)
    crossing_times = torch.cat(crossing_times_list)
    dreiss_features = torch.cat(dreiss_features_list)
    nest_ids = torch.cat(nest_id_list)
    return embeds, specs, crossing_times, nest_ids, dreiss_features


def get_all_validation_embeds(config, owlnet, owlet_dataset, collate_func):
    verification_dl = get_verification_dataloader(owlet_dataset, None, collate_func)
    validation_embeds, _, _, _, _= create_embeds(config, owlnet, verification_dl)
    validation_embeds = F.normalize(validation_embeds, p=2, dim=1)
    embeddings_2d = reduce_dimensions(validation_embeds)
    return embeddings_2d