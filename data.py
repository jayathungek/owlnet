from utils import *
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.transforms.functional import resize


# build dataset using torch
class OwletDataset(Dataset):
    def  __init__(self,
                  data_dir,
                  silence_threshold=0.5,
                  min_call_len=5, max_call_len=25,
                  test_only=False,
        ):
        super().__init__()
        self.data_dir = data_dir
        self.silence_threshold = silence_threshold
        self.min_call_len = min_call_len
        self.max_call_len = max_call_len
        self.test_only = test_only
        self.dataset = self.prepare_data()


    def prepare_data(self):
        all_wavs = list(gather_data_files(self.data_dir, test_only=self.test_only))
        chunks_list = []
        for file in all_wavs:
            print(f"Processing file {file.stem}")
            chunks, chunks_original = chop_file(
                file,
                zero_threshold=self.silence_threshold,
                min_len=self.min_call_len,
                max_len=self.max_call_len,
            )
            chunks_list += list(zip(chunks, chunks_original))

        return chunks_list

    def __len__(self):
        return len(self.dataset)
                
    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]

def get_verification_dataloader(full_dataset, verification_subset, collate_fn):
    start, end = verification_subset
    indices = list(range(start, end))
    verification_dataset = Subset(full_dataset, indices)
    dataloader = DataLoader(
        dataset=verification_dataset, 
        collate_fn=collate_fn,
        batch_size=len(indices), 
        shuffle=False
    )            
    return dataloader

class ToyDataset(Dataset):
    def  __init__(self, data_len, data_proportion, data_h, data_w):
        super().__init__()
        self.data_len = data_len
        self.data_h = data_h
        self.data_w = data_w
        self.data_proportion = data_proportion
        self.dataset = self.prepare_data()

    def prepare_data(self):
        chunks = (torch.rand(self.data_len) > self.data_proportion).int().reshape(
            self.data_len, 1).repeat(
                1, self.data_h).repeat(
                    1, self.data_w).reshape(
                        self.data_len, self.data_h, self.data_w)
        chunks = list(chunks.float().unsqueeze(1).unbind())
        return chunks
    
    def __len__(self):
        return len(self.dataset)
                
    def __getitem__(self, idx):
        return self.dataset[idx]


def safe_split(total_len, fractions):
    assert sum(fractions) == 1, f"Split fractions must sum to 1, but got: {sum(split)}"
    split = [
       round(frac * total_len)
       for frac in fractions 
    ]

    # if there is a mismatch, make up the difference by adding it to train samples
    split_sum = sum(split)
    if total_len != split_sum:
        diff = max(total_len, split_sum) - min(total_len, split_sum)
        diff = total_len - sum(split)
        split[0] += diff
    assert sum(split) == total_len, f"Expected sum of split == {total_len}, but got: {sum(split)}" 
    return split


class CollateFunc:

    def __init__(self, spec_height):
        self.spec_height = spec_height
    
    def __call__(self, batch):
        max_len = 0
        for spec, og_spec  in batch:
            time_len = spec.shape[-1]
            max_len = time_len if time_len > max_len else max_len
            
        
        padded = []
        padded_og = []
        for spec, og_spec in batch:
            padding_needed = max_len - spec.shape[2]
            if padding_needed > 0:
                # spec = F.pad(spec, (0, padding_needed), "constant", 0)
                # og_spec = F.pad(og_spec, (0, padding_needed), "constant", 0)

                spec = resize(spec, (self.spec_height, max_len), antialias=True)
                og_spec = resize(og_spec, (self.spec_height, max_len), antialias=True)

            padded.append(spec)
            padded_og.append(og_spec)
            
        padded = torch.cat(padded).unsqueeze(1)
        padded_og = torch.cat(padded_og).unsqueeze(1)
        return padded, padded_og


def load_data(
    data_dir, 
    spec_height,
    batch_sz=16,
    train_test_split=[0.8, 0.2],
    debug=False,
):

    assert sum(train_test_split) == 1, "Train and test fractions should sum to 1!"  
    dataset = OwletDataset(
        data_dir,
        silence_threshold=0.7,
        min_call_len=10, max_call_len=200,
        test_only=debug,
    )
    
    # This code generates the actual number of items that goes into each split using the user-supplied fractions
    tr_te = safe_split(len(dataset), train_test_split)
    train_split, test_split = random_split(dataset, tr_te)

    collate_func = CollateFunc(spec_height=spec_height)
    
    train_dl = DataLoader(
        train_split, 
        collate_fn=collate_func,
        batch_size=batch_sz, 
        shuffle=True, 
    )            

    test_dl = DataLoader(
        test_split, 
        collate_fn=collate_func,
        batch_size=batch_sz, 
        shuffle=False, 
    )

    return train_dl, test_dl, dataset



def load_toy_data(
    batch_sz=16,
    train_test_split=[0.8, 0.2]
):
    def collate_func(batch):
        padded = []
        padded_og = []
        for spec, spec_og in batch:
            padded.append(spec)
            padded_og.append(spec_og)
        padded = torch.cat(padded).unsqueeze(1)
        padded_og = torch.cat(padded_og).unsqueeze(1)
        return padded, padded_og

    assert sum(train_test_split) == 1, "Train and test fractions should sum to 1!"  
    # dataset = OwletDataset(data_dir, silence_threshold=0.7, min_call_len=10, max_call_len=200)
    dataset = ToyDataset(2000, 0.2, 256, 70)
    
    # This code generates the actual number of items that goes into each split using the user-supplied fractions
    tr_te = safe_split(len(dataset), train_test_split)
    train_split, test_split = random_split(dataset, tr_te)
    
    train_dl = DataLoader(
        train_split, 
        collate_fn=collate_func,
        batch_size=batch_sz, 
        shuffle=True, 
    )            

    test_dl = DataLoader(
        test_split, 
        collate_fn=collate_func,
        batch_size=batch_sz, 
        shuffle=False, 
    )

    return train_dl, test_dl