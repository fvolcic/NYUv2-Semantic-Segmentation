import torch
import torch.nn as nn
import utils.utils as utils
#from segmentation_models_pytorch.losses import DiceLoss
from dataloaders.NYU_loader import NYUv2
from models.enet import enet as enet
from models.basic_unet import Unet
from tqdm import tqdm

MODEL_OUTPUT_PATH = "trained_models/enet_depth_nyuv2_attempt2/"
NUM_EPOCHS = 10000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

model = enet(num_classes=14, in_channels=4).to(device) # 14 classes, 4 channels
model.train()

loss_func_1 = utils.DiceLoss()
loss_func_2 = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    'min', 
    factor=0.5, 
    patience=10, 
    min_lr=0.00001,
    verbose=True)

train_dl = NYUv2(
    root="content/NYUv2/",
    train=True, 
    download=False, 
    img_size=(512, 512), 
    load_rgb=True, 
    load_depth=True, 
    load_seg=True, 
    load_sn=False,
    random_crop=False,
    random_flip=True,
)

train_loader = torch.utils.data.DataLoader(
    dataset = train_dl,
    batch_size = 16,
    shuffle = True,
    num_workers = 4)

losses_train = [] # losses for each epoch

for epoch in range(NUM_EPOCHS):
    print("epoch: ", epoch)
    losses_epoch = [] # losses for this epoch
    dice_loss_epoch = []
    cross_entropy_loss_epoch = []
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        
        rgb = data[0].to(device) # rgb
        depth = data[2].to(device) # depth
        
        y = data[1].to(device) # segmentation
        y = utils.convert_to_one_hot(y, classes=14, scale_values=False)

        # load rgb and depth, then concatinate them
        rgb = rgb.to(device)
        depth = depth.to(device)
        x = torch.cat((rgb, depth), dim=1)
        
        y_pred = model(x)
        
        loss1 = loss_func_1(y_pred, y)
        loss2 = loss_func_2(y_pred, y)
        loss = loss1 + loss2
        loss.backward()
         
        losses_epoch.append(loss.detach().item())
        dice_loss_epoch.append(loss1.detach().item())
        cross_entropy_loss_epoch.append(loss2.detach().item())
        optimizer.step()
    
    epoch_loss = torch.mean(torch.tensor(losses_epoch))
    epoch_dice_loss = torch.mean(torch.tensor(dice_loss_epoch))
    epoch_cross_entropy_loss = torch.mean(torch.tensor(cross_entropy_loss_epoch))
    scheduler.step(epoch_loss) 
    print("epoch loss: ", epoch_loss.item())
    print("epoch dice loss: ", epoch_dice_loss.item())
    print("epoch cross entropy loss: ", epoch_cross_entropy_loss.item())

    if epoch % 1 == 0:
        # now save the model and the losses
        torch.save(model.state_dict(), MODEL_OUTPUT_PATH + f"enet_epoch_{epoch}.pth")

        # save the losses
        with open(MODEL_OUTPUT_PATH + f"epoch_{epoch}_losses.txt", "w") as f:
            for loss in losses_epoch:
                f.write(f"{loss},")
        with open(MODEL_OUTPUT_PATH + f"epoch_{epoch}_dice_losses.txt", "w") as f:
            for loss in dice_loss_epoch:
                f.write(f"{loss},")
        with open(MODEL_OUTPUT_PATH + f"epoch_{epoch}_cross_entropy_losses.txt", "w") as f:
            for loss in cross_entropy_loss_epoch:
                f.write(f"{loss},")
