
from torch import nn
import torch

from tqdm import tqdm

from utils.utils import show_result
import utils.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_Unet(model, dataloader, optimizer=None, loss_func=None, num_epochs=20, show_result_every=5):
    
    total_losses = []

    for epoch in range(num_epochs):
        epoch_losses = []

        for data in tqdm(dataloader):
            x = data[0].to(device=device).float()
            y = data[1].to(device=device).float()

            y = utils.convert_to_one_hot(y, 13)

            optimizer.zero_grad()
            output = model(x)

            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            total_losses.append(loss.item())

        epoch_losses = torch.tensor(epoch_losses)
        
        print(f"Epoch {epoch} loss: {torch.sum(epoch_losses)/(epoch_losses.shape[0])}")
    
    return total_losses

