
import torch
from tqdm import tqdm

from ...constants import DEVICE


def process_segmentation(segmentation):
    segmentation = torch.round(segmentation * 12).int()
    return segmentation


def validate(model, dataloader, num_batches=None):
    loss_epoch = 0
    corrects_epoch = 0
    print("Validating...")
    batch = 0

    if not num_batches is None:
        num_batches = len(dataloader)

    class BatchLoader:
        def __init__(self, dataloader):
            self.dataloader = dataloader
            self.batch = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.batch == num_batches:
                raise StopIteration
            self.batch += 1
            return next(self.dataloader)

    loader = BatchLoader(dataloader)

    for data in tqdm(loader):

        if batch == num_batches:
            break
        batch += 1

        x = data[0].to(device=DEVICE)
        y = process_segmentation(data[1].to(device=DEVICE)).to(device=DEVICE)
        output = model(x).to(device=DEVICE)
        output = process_segmentation(output).to(device=DEVICE)

        corrects_epoch += torch.sum((output - y) < 1e-3)

    total_pixels = batch * torch.numel(output)
    epoch_acc = corrects_epoch / total_pixels

    print('Test accuracy {}'.format(epoch_acc))

