import os
import re
import torch

def find_latest_checkpoint(checkpoint_dir, model_name, backbone):
    pattern = re.compile(rf'{model_name}_{backbone}_([0-9]+)_([0-9\.]+)\.pth$')
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            epoch, dice_score = map(float, match.groups())
            checkpoints.append((epoch, dice_score, filename))
    if checkpoints:
        # Sort by epoch
        latest_checkpoint = sorted(checkpoints, key=lambda x: x[0], reverse=True)[0]
        return latest_checkpoint
    return None
