from pathlib import Path

import torch
import torchaudio
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm.notebook import tqdm
import torchvision.transforms as trans


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
    

def get_melspec(waveform, max_freq=750):
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=4096,
        hop_length=1024,
        power=2.0
    )
    mel_spec = spectrogram(waveform)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mel_spec = normalise(mel_spec)[:, :max_freq, :]
    return mel_spec


def display_melspec(melspec, crossings=None, size=(10, 4), colorbar=True):
    plt.figure(figsize=size)
    melspec = melspec.squeeze().numpy()
    plt.imshow(melspec, aspect='auto', origin='lower', cmap='viridis')

    if crossings is not None:
        for idx in crossings:
            plt.axvline(x=idx, color='r', linestyle='-', linewidth=0.5)
    if colorbar:
        plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show() 


def display_audio_file(wav_path):
    waveform, _ = torchaudio.load(wav_path)
    plt.figure(figsize=(10, 4))
    plt.plot(waveform.t().numpy())  # Convert PyTorch tensor to NumPy
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
    mel_spec_db = get_melspec(waveform)
    display_melspec(mel_spec_db)


def gather_data_files(data_dir, test_only):
    pattern = 'test.wav' if test_only else '1*.wav'
    data_dir = Path(data_dir).resolve()
    all_files = data_dir.glob(pattern)
    return all_files
    

def get_zero_crossing_indices(melspec, zero_threshold, min_len, max_len):
    # replace all values below threshold with 0 and average along the 
    # frequency axis
    mean_freq_axis = melspec.mean(dim=1).squeeze()
    filtered = torch.where(mean_freq_axis<= zero_threshold, 0, mean_freq_axis)

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
    


def show_batch(image_batch, title="Image batch", size=10):
    result = image_grid(image_batch)
    plt.figure(figsize=(size, size))
    plt.title(f"{title}")
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.gca().spines['top'].set_visible(False)   # Remove top border
    plt.gca().spines['right'].set_visible(False) # Remove right border
    plt.gca().spines['left'].set_visible(False)  # Remove left border
    plt.gca().spines['bottom'].set_visible(False) # Remove bottom border
    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    plt.imshow(result, interpolation="nearest")
    
    
    
    
def chop_file(filepath,
              zero_threshold=0.7,
              min_len=5, max_len=200,
              display=False,
    ):
    waveform, _ = torchaudio.load(filepath)
    melspec = get_melspec(waveform)
    chunk_indices = get_zero_crossing_indices(melspec, zero_threshold, min_len, max_len)
    chunks_normal = []
    chunks = []
    for i in tqdm(range(0, len(chunk_indices), 2)):
        start = chunk_indices[i]
        end = chunk_indices[i + 1]
        chunk = melspec[:, :, start:end]
        chunks_normal.append(chunk)
        chunk = process_melspec(chunk)
        chunks.append(chunk)
    if display:
        display_melspec(melspec, chunk_indices)
    return chunks, chunks_normal

    
def display_zero_crossings():
    all_wavs = list(gather_data_files("lotek_owl_data", test_only=True))
    chunks_list = []
    for idx, file in enumerate(all_wavs):
        print(f"Processing file {file}")
        chunks, original_chunks = chop_file(file, display=True)
        chunks_list += chunks
        
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
    