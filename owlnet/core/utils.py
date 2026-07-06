import io
import json
import colorsys
from pathlib import Path
from datetime import datetime

import numpy as np
from umap import UMAP
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torchaudio
import torch.nn.functional as F
import torchvision.transforms as trans
import torchaudio.functional as A
import pcen

try:
    from owlnet.core.model import OwlNet
except ModuleNotFoundError:
    from model import OwlNet


try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is None:
        # we have IPython installed but not running from IPython
        from tqdm import tqdm
    else:
        from tqdm.notebook import tqdm
except:
    # We do not even have IPython installed
    from tqdm import tqdm



def normalise(tensor):
    mean = tensor.mean()
    std = tensor.std()
    norm = (tensor - mean) / (std + 1e-8)
    return norm

def normalise_minmax(tensor):
    minimum = tensor.min()
    maximum = tensor.max()

    if minimum == maximum:
        return torch.zeros_like(tensor)
    
    return (tensor - minimum) / (maximum - minimum)


def process_melspec(melspec):
    enhanced_spec = F.softmax(melspec, dim=1) 
    norm = normalise_minmax(enhanced_spec)
    return norm
    

def get_melspec(config, waveform, sr):
    hipassed = hipass(waveform, sr, config['hipass_cutoff_hz'])
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        power=1.0
    )
    mel_spec = spectrogram(hipassed).permute(0, 2, 1)
    normalised, _ = pcen.pcen(
        mel_spec,
        s=config["pcen_s"],
        alpha=config["pcen_alpha"],
        delta=config["pcen_delta"],
        r=config["pcen_r"]
    )
    return normalised.permute(0, 2, 1)


def loudness_deviation(spec):
    spec = spec.squeeze()
    energy = spec.sum(dim=0)
    mid = len(energy) // 2
    first_half = energy[: mid].sum()
    second_half = energy[mid:].sum()
    deviation = second_half / (first_half + second_half)
    return deviation
    

def mean_spec_freq(spec, sr, n_fft):
    spec = spec.squeeze()
    freq_energy = spec.sum(dim=1)
    freqs = torch.arange(spec.shape[0])
    freqs = (freqs * sr) / n_fft
    mean_freq = torch.sum(freqs * freq_energy) / freq_energy.sum()
    normed = mean_freq / (sr / 2) # nyquist freq
    return normed 


def upper_freq(spec, sr, n_fft):
    spec = spec.squeeze()
    freq_energy = spec.sum(dim=1)
    distribution = torch.cumsum(freq_energy, dim=0)
    distribution = distribution / distribution[-1]

    idx = torch.searchsorted(distribution, 0.75)
    upper = (idx * sr) / n_fft
    normed = upper / (sr / 2) # nyquist freq
    return normed


def freq_variation(spec, sr, n_fft):
    spec = spec.squeeze()
    frame_centroids = []
    freqs = torch.arange(spec.shape[0])
    freqs = (freqs * sr) / n_fft
    for t in range(spec.shape[1]):
        spectrum = spec[:, t]
        energy = spectrum.sum()
        if energy > 0:
            centroid = torch.sum(freqs * spectrum) / energy
            frame_centroids.append(centroid)
    freq_variation = torch.std(torch.tensor(frame_centroids))
    normed = freq_variation / (sr / 2) # nyquist freq
    return normed


def display_melspec(melspec, crossings=None, size=(20, 4), colorbar=True):
    plt.figure(figsize=size)
    melspec = melspec.squeeze().numpy()
    plt.imshow(melspec, aspect='auto', origin='lower', cmap='viridis')

    if crossings is not None:
        if type(crossings) is list:
            crossings = torch.tensor(crossings).flatten()
        for idx in crossings:
            plt.axvline(x=idx, color='r', linestyle='-', linewidth=0.5)
    if colorbar:
        plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show() 


def display_audio_file(config, wav_path):
    waveform, sr = torchaudio.load(wav_path)
    plt.figure(figsize=(10, 4))
    plt.plot(waveform.t().numpy())  # Convert PyTorch tensor to NumPy
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

    
    mel_spec_db = get_melspec(config, waveform, sr)
    display_melspec(mel_spec_db)


def gather_data_files(config):
    data_dir = Path(config["data_dir"]).resolve()
    test_only = config["debug"]
    if test_only:
        all_files = data_dir.glob(config["test_file"])
    else:
        mp3_files = [f for f in data_dir.glob("*.mp3") if f.name != config["test_file"]]
        wav_files = [f for f in data_dir.glob("*.wav") if f.name != config["test_file"]]
        all_files = mp3_files + wav_files
    return all_files


def get_zero_crossing_indices(melspec, zero_threshold, min_len, max_len, display):
    # replace all values below threshold with 0 and average along the 
    # frequency axis
    mean_freq_axis = melspec.mean(dim=1).squeeze()
    # mean_freq_axis /= torch.max(mean_freq_axis)
    mean_freq_axis = (mean_freq_axis - mean_freq_axis.min()) / (mean_freq_axis.max() - mean_freq_axis.min())

    if display:
        plt.plot(mean_freq_axis)
        plt.axhline(y=zero_threshold, color="red")

    filtered = torch.where(mean_freq_axis<= zero_threshold, 0, 1)

    # plt.figure(figsize=(12, 2))
    # plt.imshow(filtered[np.newaxis, :], aspect='auto', interpolation='nearest')
    # plt.yticks([])            
    # plt.xlabel("Index")
    # plt.title("Boolean array (True/False)")
    # plt.tight_layout()
    # plt.show()

    # Use 2 pointers to keep track of start and end of a call
    crossings = []
    start_pointer = 0
    end_pointer = 0
    while end_pointer < filtered.shape[0]:
        start_in_call = filtered[start_pointer] > 0
        end_in_call = filtered[end_pointer] > 0
        if start_in_call != end_in_call:
            # we have crossed over to another state, calculate length 
            # of the previous state and check what it is. If it is a
            # call and fits length criteria, add pointers to 
            # crossing list
            state_len = end_pointer - start_pointer
            if start_in_call and (min_len < state_len < max_len):
                crossings += [start_pointer, end_pointer] 
            start_pointer = end_pointer
        end_pointer += 1
    return crossings
    


def image_grid(image_batch):
    batch_sz, height, width = image_batch.shape
    image_batch = image_batch.unsqueeze(-1)
    image_batch = image_batch.permute(1, 0, 2, 3)
    grid = image_batch.reshape(height, width * batch_sz)

    spacer_width = 50
    num_spacers = batch_sz - 1
    grid_with_spacers = []
    for n in range(batch_sz):
        grid_with_spacers.append(
            grid[:, (width * n) : (width * n) + width]
        )
        if n < batch_sz - 1:
            grid_with_spacers.append(
                torch.zeros(height, spacer_width)
            )
    grid_with_spacers = torch.cat(grid_with_spacers, dim=1)
    grid = grid_with_spacers.reshape(height, (width * batch_sz) + (num_spacers * spacer_width))
    return grid
    


def show_batch(image_batch, title="Image batch", size=(20, 4)):
    result = image_grid(image_batch)
    plt.figure(figsize=size)
    plt.title(f"{title}")
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.gca().spines['top'].set_visible(False)   # Remove top border
    plt.gca().spines['right'].set_visible(False) # Remove right border
    plt.gca().spines['left'].set_visible(False)  # Remove left border
    plt.gca().spines['bottom'].set_visible(False) # Remove bottom border
    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    plt.imshow(result, interpolation="nearest")
    
    
    
    
def chop_file(
    config,
    filepath,
    t_init,
    display=False,
):
    waveform, sample_rate = torchaudio.load(filepath)
    melspec = get_melspec(config, waveform, sample_rate)
    hop_size = config['hop_length']
    min_len = int(((config['min_call_len_ms'] / 1000) * sample_rate) / hop_size)
    max_len = int(((config['max_call_len_ms'] / 1000) * sample_rate) / hop_size)
    chunk_indices = get_zero_crossing_indices(
        melspec,
        config['zero_threshold'],
        min_len,
        max_len,
        display=display
    )
    chunks = []
    chunks_crossing_times = []
    for i in tqdm(range(0, len(chunk_indices), 2)):
        start = chunk_indices[i]
        end = chunk_indices[i + 1]
        start_time = start * (hop_size / sample_rate)
        end_time = end * (hop_size / sample_rate)
        chunk = melspec[:, :, start:end]
        # chunk = process_melspec(chunk)
        chunks.append(chunk)
        chunks_crossing_times.append([t_init + start_time, t_init + end_time])
    if display:
        display_melspec(melspec, chunk_indices)
    return chunks, chunks_crossing_times, sample_rate

    
def display_zero_crossings(config, display=True):
    all_wavs = list(gather_data_files(config))
    chunks_list = []
    chunk_crossing_times_list = []
    for idx, file in enumerate(all_wavs):
        print(f"Processing file {file}")
        chunks, original_chunks, chunk_crossing_times = chop_file(config, file, display=True)
        chunks_list += chunks
        chunk_crossing_times_list.append(chunk_crossing_times)
        
    if display:
        max_time = 0
        height = chunks_list[0].shape[1]
        for spec in chunks_list:
            t = spec.shape[-1]
            if t > max_time:
                max_time = t

        resize = trans.Resize((height, max_time), antialias=True)
        chunks_list = [
            resize(
                torch.cat(
                    list(reversed(
                        c.unbind(dim=1)
                    ))
                ).unsqueeze(0)
            ) for c in chunks_list
        ]
        spectrograms = torch.cat(chunks_list)
        show_batch(spectrograms, title="Model inputs")
    return chunks, original_chunks, chunk_crossing_times
    



def imshow_to_pil(image_array, cmap="viridis"):
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.axis("off")  # Remove axes
    image_array = image_array.squeeze()
    ax.imshow(image_array, cmap=cmap, aspect="auto", origin="lower")

    # Save figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Convert buffer to PIL Image
    buf.seek(0)
    pil_image = Image.open(buf)
    
    return pil_image

def reduce_dimensions(embeddings):
    # reducer = PCA(n_components=2, svd_solver='auto')
    # points = reducer.fit_transform(embeddings)

    reducer = UMAP(n_components=2, metric="cosine")#, random_state=0, transform_seed=0)
    reducer.fit(embeddings)
    points = reducer.transform(embeddings)
    return points


def get_label_colours(n):
    colors = []
    hue_step = 360.0 / n

    for i in range(n):
        hue = i * hue_step
        saturation = 1  # You can adjust saturation and lightness if needed
        lightness = 0.4   # You can adjust saturation and lightness if needed

        rgb = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)
        hexcol = "#" + "".join([f"{int(v * 255):02X}" for v in rgb])
        colors.append(hexcol)

    return colors
    

def load_config(config_path):
    with open(config_path) as fh:
        config_dict = json.load(fh)
    return config_dict

    
def get_model(config, model_name=None):
    drop = config["drop"]
    embed_sz = config["embed_sz"]
    enc_out_dim = config["enc_out_dim"]
    device = config["device"]
    checkpoint_dir = config["checkpoint_dir"]
    attention = config["use_attn"]
    num_dreiss_features = config["num_dreiss_features"]

    if model_name is None:
        model_name = config["default_model"]

    model_dir = f"{config['proj_root']}/{checkpoint_dir}/{model_name}"
    best_checkpoint = get_sorted_checkpoints(model_dir)[0]
    print(f"Got best checkpoint from {model_name}: {best_checkpoint.stem}")

    save_items = torch.load(best_checkpoint, map_location=torch.device(device))
    owlnet_dict = toggle_model_dict_dataparallel(save_items["model_state_dict"])
    if device == "cuda":
        owlnet = nn.DataParallel(OwlNet(
            enc_out_dim, 
            embed_sz,
            drop,
            num_dreiss_features,
            use_attention=attention
        )).to(device)
        owlnet.module.load_state_dict(owlnet_dict)
    else:
        owlnet = OwlNet(
            enc_out_dim,
            embed_sz,
            drop,
            num_dreiss_features,
            use_attention=attention
        ).to(device)
        owlnet.load_state_dict(owlnet_dict)
    return owlnet
        

def get_img_data(img_path):
    with open(img_path, "rb") as fh:
        data = fh.read()
    return data

    
def hipass(signal, sr, cutoff_hz):
    filtered = A.highpass_biquad(signal, sample_rate=sr, cutoff_freq=cutoff_hz)
    return filtered

    
def infer_abs_unix_timestamp(filename):
    time_part = filename.split("_")[-1]
    date_part = filename.split("_")[-2]
    date_time = f"{date_part}{time_part}"
    ts = datetime.strptime(date_time, "%Y%m%d%H%M%S").timestamp()
    return ts


def display_datetime(timestamp):
    return (
        datetime.fromtimestamp(timestamp)
    )


def get_nest_id(path):
    prefix = "nest"
    path_tags = str(path).split("_")
    nest_num = None
    for tag in path_tags:
        if prefix in tag:
            idx = tag.index(prefix)
            nest_num = int(tag[idx+len(prefix):])
    if nest_num is None:
        assert False, f"Filenames incorrect or not found: must include `nest` in title. Check data root dir"
    return nest_num


def get_sorted_checkpoints(model_dir: str, sortby="metric", epoch=None):
    model_dir = Path(model_dir).resolve()
    if epoch is not None:
        pattern = f'epoch_{epoch}*.pth'
        ret_checkpoints = list(model_dir.glob(pattern))
        return ret_checkpoints

    all_checkpoints = list(model_dir.glob('*.pth'))
    get_value = lambda x: float(x.stem.split('_')[-1]) 
    if sortby == "metric":
        if len(all_checkpoints) > 0:
            sorted_checkpoints =  sorted(
                all_checkpoints,
                key=get_value,
            )
            return sorted_checkpoints
        else:
            return []
    else:
        # sortby should contain the exact value that needs to be 
        # used.
        min_dist = float("inf")
        sorted_checkpoints = None
        for p in all_checkpoints:
            v = get_value(p)
            dist = abs(v - sortby)
            if dist < min_dist:
                min_dist = dist
                if sorted_checkpoints is None:
                    sorted_checkpoints = [p]
                else:
                    sorted_checkpoints[0] = p
        if sorted_checkpoints is None:
            sorted_checkpoints = []
        return sorted_checkpoints


def toggle_model_dict_dataparallel(model_dict):
    dict_keys = list(model_dict.keys())
    dp_prefix = "module."
    is_data_parallel = dict_keys[0].startswith(dp_prefix)

    if is_data_parallel:
        ret_dict = {
            k[len(dp_prefix):]: v 
            for k, v in model_dict.items()
        }
    else:
        ret_dict = {
            f"{dp_prefix}{k}": v 
            for k, v in model_dict.items()
        }

    return ret_dict


def save_model(run_path, curr_epoch, model, opt, scaler, loss=None):
    metric_str = f"loss_{loss}"
    with open(run_path / f"epoch_{curr_epoch}_{metric_str}.pth", "wb") as fh:
        if not isinstance(model, dict):
            model = model.state_dict()
        if not isinstance(opt, dict):
            opt = opt.state_dict()
        if not isinstance(scaler, dict):
            scaler = scaler.state_dict()
            
        save_items = {
            "epoch": curr_epoch,
            "model_state_dict": model,
            "optimizer_dict": opt,
            "scaler_dict": scaler,
        }
        torch.save(save_items, fh)


if __name__ == "__main__":
    # print(infer_abs_unix_timestamp("2MM09330_20250625_033002"))
    # p = Path("owl_data/nest3_smu2_trigon_2025/2MM09330_20250625_003002.wav")
    # print(get_nest_id(p))
    num_files = len(list(Path("owl_data/").glob("**/*.[Ww][Aa][Vv]")))
    print(num_files)