import os
from datetime import datetime
import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import time
from configurations import Config
from src.utilities.accuracy import AccuracyTracker
from src.utilities.checkpoint import find_latest_checkpoint
from src.utilities.early_stopping import EarlyStopping
from src.utilities.log import log_to_file
from src.dataset.uav_dataset import UAVDataset

accuracyTrackerTrain = AccuracyTracker(n_classes=14)
accuracyTrackerVal = AccuracyTracker(n_classes=14)

colors = ['green', 'red', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'pink', 'lime', 'brown', 'gray',
          'olive', 'teal', 'navy']
cmap = ListedColormap(colors[:15])

IMG_SIZE = 512
MODEL_NAME = 'PSPNET'
BACKBONE = 'mobilenet'
ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
# IMG_SIZE = 256
# MODEL_NAME = 'UNET'
# MODEL_NAME = 'FPN'
# BACKBONE = 'efficientnet-b0'
# ENCODER = 'efficientnet-b0'
N_CLASSES = 14
ACTIVATION = 'sigmoid'
# ACTIVATION = 'softmax'

mean = [0.4607, 0.4558, 0.4192]
std = [0.2200, 0.2067, 0.2227]


def train(model, config, train_loader):
    model.train()
    model.to(config.device)
    running_loss = 0
    iteration = 0

    loop = tqdm(train_loader)

    for batch_idx, (inputs, labels) in enumerate(loop):
        iteration += 1
        inputs, labels = inputs.to(config.device), labels.to(config.device)

        config.optimizer.zero_grad()
        outputs = model(inputs)
        loss = config.criterion(outputs, labels)
        loss.backward()
        config.optimizer.step()

        loop.set_postfix(loss=loss.item())
        running_loss += loss.item()

        outputs = outputs.cpu().data.max(1)[1].numpy()
        labels = labels.cpu().data.max(1)[1].numpy()
        outputs.astype(np.uint8)
        labels.astype(np.uint8)
        accuracyTrackerTrain.update(labels, outputs)

    train_loss = running_loss / iteration
    config.scheduler.step()
    print('Train Loss: %.3f' % train_loss)
    return train_loss


def evaluation(model, config, val_loader):
    model.eval()

    running_loss = 0
    running_time = 0
    iteration = 0

    saved_images = np.zeros((3, IMG_SIZE, IMG_SIZE, 3))

    loop = tqdm(val_loader)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loop):
            iteration += 1
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            running_time += end_time - start_time

            loss = config.criterion(outputs, labels)
            running_loss += loss.item()

            outputs = outputs.cpu().data.max(1)[1].numpy()
            labels = labels.cpu().data.max(1)[1].numpy()

            outputs.astype(np.uint8)
            labels.astype(np.uint8)

            accuracyTrackerVal.update(labels, outputs)

            if batch_idx == 0:
                saved_images[0] = np.transpose(inputs.cpu().numpy()[0], (1, 2, 0))
                label = labels[0].reshape(IMG_SIZE, IMG_SIZE, 1)
                output = outputs[0].reshape(IMG_SIZE, IMG_SIZE, 1)
                saved_images[1] = cmap(np.repeat(label[:, :, np.newaxis], 3, axis=2).reshape(IMG_SIZE, IMG_SIZE, 3))[:,
                                  :, 0, :3]
                saved_images[2] = cmap(np.repeat(output[:, :, np.newaxis], 3, axis=2).reshape(IMG_SIZE, IMG_SIZE, 3))[:,
                                  :, 0, :3]

    val_loss = running_loss / iteration
    val_time = running_time / iteration
    print('Eval Loss: %.3f' % val_loss)
    return val_loss, val_time, saved_images


def main():
    config = Config()

    # Model and transformations setup
    # model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=N_CLASSES).to(config.device)
    model = smp.FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=N_CLASSES).to(config.device)
    # model = smp.PSPNet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=N_CLASSES).to(config.device)
    transform = A.Compose([A.Resize(width=IMG_SIZE, height=IMG_SIZE, interpolation=cv2.INTER_NEAREST)])

    # Checkpoint loading logic
    checkpoint_dir = '//checkpoint/'
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir, MODEL_NAME, BACKBONE)
    if latest_checkpoint:
        epoch, best_dice, filename = latest_checkpoint
        epoch = int(epoch)
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        model = torch.load(checkpoint_path, map_location=config.device)
        print(f"Resuming from epoch {epoch} with best Dice {best_dice} from checkpoint {filename}")
    else:
        epoch = 0
        best_dice = 0
        print("No checkpoint found, starting training from scratch.")

    # Augmentations
    aug_data = A.Compose(
        [A.Resize(width=IMG_SIZE, height=IMG_SIZE, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(p=0.5),
         A.VerticalFlip(p=0.5), A.Rotate(limit=[-60, 60], p=0.8, interpolation=cv2.INTER_NEAREST),
         A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.2], contrast_limit=0.2, p=0.3)], p=1.0)

    # Dataset setup
    train_dataset = UAVDataset(datapath=config.datapath, transform=aug_data, mean=mean, std=std, train=True)
    val_dataset = UAVDataset(datapath=config.datapath, transform=transform, mean=mean, std=std, train=False)

    # Data loader setup
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Loss, optimizer, and scheduler setup
    config.set_model_related_configs(model)

    # Log setup
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))

    early_stopping = EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True)

    for epoch in range(epoch+1, config.epochs + 1):
        # Reset accuracy trackers, etc.
        print(f'\nEpoch: {epoch}')
        log_to_file(log_dir, f'Epoch: {epoch}')

        train_loss = train(model, config, train_loader)
        val_loss, val_time, saved_images = evaluation(model, config, val_loader)

        input_image, target_image, pred_image = saved_images[0], saved_images[1], saved_images[2]

        # Update log
        log_to_file(log_dir,
                    f'Epoch: {epoch}, '
                    f'Training Loss: {train_loss}, '
                    f'Validation Loss: {val_loss}, '
                    f'Accuracy Train: {accuracyTrackerTrain.get_accuracy()}, '
                    f'Accuracy Val: {accuracyTrackerVal.get_accuracy()}, '
                    f'Mean Dice Train: {accuracyTrackerTrain.get_mean_dice()}, '
                    f'Mean Dice Val: {accuracyTrackerVal.get_mean_dice()}, '
                    f'Inference Time: {val_time}, '
                    f'Learning Rate: {config.scheduler.get_last_lr()[-1]}')

        if accuracyTrackerVal.get_mean_dice() > best_dice:
            torch.save(model, '/Users/bob/PycharmProjects/UAV-2023/checkpoint/' + MODEL_NAME + '_' + BACKBONE + '_' + str(epoch) + '_' + str(accuracyTrackerVal.get_mean_dice()) + '.pth')
            best_dice = accuracyTrackerVal.get_mean_dice()

        # Check for early stopping
        if early_stopping(model, val_loss):
            print("Early stopping triggered.")
            print(early_stopping.status)
            break  # Exit the training loop

    # Save the best model if early stopping was triggered
    if early_stopping.best_model is not None:
        torch.save(early_stopping.best_model, '/checkpoint/PSPNET_mobilenet_best_model.pth')


if __name__ == '__main__':
    main()
