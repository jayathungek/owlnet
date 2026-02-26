from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from owlnet.core.model import OwlNet
from owlnet.core.losses import ContrastiveLoss
from owlnet.data.dataloading import load_data
from owlnet.core.utils import (
    save_model,
    get_sorted_checkpoints,
    toggle_model_dict_dataparallel
)


def train(config, model_name):
    load_obj = load_data(config)
    owlet_train = load_obj["train"]["dl"]
    train_ds_size = load_obj["train"]["size"]
    model_name = f"{model_name}_{config['train_epochs']}.epochs"
    owlnet = OwlNet(
        config['embed_sz'],
        config['drop'],
        config['use_attn']
    ).to(config["device"])
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer = torch.optim.AdamW(owlnet.parameters(), betas=(0.9, 0.999), lr=config["learning_rate"])
    curr_epoch = 0
    run_path = Path(config['checkpoint_dir']) / model_name

    if not run_path.exists():
        run_path.mkdir(exist_ok=True)
    else:
        all_checkpoints = get_sorted_checkpoints(run_path)
        if len(all_checkpoints) > 0:
            best_chkpt = all_checkpoints[0]
            print(f"Run exists, loading from checkpoint: {best_chkpt}")
            save_items = torch.load(best_chkpt)
            model_dict =  save_items["model_state_dict"]
            model_dict = toggle_model_dict_dataparallel(model_dict)
            owlnet.load_state_dict(model_dict)
            optimizer.load_state_dict(save_items["optimizer_dict"])
            scaler.load_state_dict(save_items["scaler_dict"])
            curr_epoch = save_items["epoch"] + 1
    
    owlnet = nn.DataParallel(owlnet, device_ids=[0, 1])
    loss_function = ContrastiveLoss()
    owlnet.train()
    try:
        pbar = tqdm(range(curr_epoch, config["train_epochs"]))
        for epoch in pbar:
            num_batches = train_ds_size / config["batch_sz"]
            total_loss = 0
            total_batches = 0
            for i, (train_batch, train_aug_batch, _, _) in enumerate(owlet_train):
                optimizer.zero_grad()
                train_batch = train_batch.to(config["device"])
                train_aug_batch = train_aug_batch.to(config["device"])
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True), torch.backends.cuda.sdp_kernel(enable_flash=False):
                    embeds = owlnet(train_batch)
                    embeds_aug = owlnet(train_aug_batch)
                    embeds_cat = torch.cat(
                        (
                            F.normalize(embeds, p=2, dim=1).unsqueeze(1),
                            F.normalize(embeds_aug, p=2, dim=1).unsqueeze(1)
                        ), dim=1
                    )
                    loss = loss_function(embeds_cat)
                    total_loss += loss
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                pbar.set_description(f"Train {epoch+1} | {100*i/num_batches:.3f}% | Loss: {loss.item():.4f}")
                total_batches += 1
            
            total_loss /= total_batches
            if epoch % config["checkpoint_freq"] == 0:
                save_model(run_path, epoch, owlnet, optimizer, scaler, total_loss.item())
    except KeyboardInterrupt:
        print(f"Training interrupted by user, saving latest model weights")
        save_model(run_path, epoch, owlnet, optimizer, scaler, loss.item())


if __name__ == "__main__":
    config = "settings/config.json"
    train(config, "stoke_wake_all")
