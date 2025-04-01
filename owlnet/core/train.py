import torch
import torch.nn.functional as F

from owlnet.core.losses import loss_func
from owlnet.core.model import OwlNet
from owlnet.data.dataloading import load_data


def train(config, model_name):
    owlet_train, _, dataset = load_data(config)
    model_name = f"{model_name}_{len(dataset)}.datapoints_{config['train_epochs']}.epochs"
    owlnet = OwlNet(
        config['embed_sz'],
        config['drop'],
        config['use_attn']
    ).to(config["device"])
    optimizer = torch.optim.AdamW(owlnet.parameters(), betas=(0.9, 0.999), lr=config["learning_rate"])
    loss_function = loss_func
    owlnet.train()
    for epoch in range(config["train_epochs"]):
        for i, (train_batch, _, _) in enumerate(owlet_train):
            optimizer.zero_grad()
            train_batch = train_batch.to(config["device"])
            embeds = owlnet(train_batch)
            embeds = F.normalize(embeds, p=2, dim=1) 
            loss = loss_function(embeds)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch {epoch + 1} ({i}/{len(owlet_train)}): Loss {loss.item()}",
                end="\r" if epoch < config['train_epochs'] - 1 else "\n"
            )
        
    torch.save(owlnet.state_dict(), f"{config['checkpoint_dir']}/{model_name}.pth")


