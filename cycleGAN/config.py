import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "mps"
TRAIN_DIR = "horse2zebra/train/"
VAL_DIR = "horse2zebra/val/"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
LAMBDA_IDENTITY= 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC_H = "models/disch.pth.tar"
CHECKPOINT_DISC_Z = "models/discz.pth.tar"
CHECKPOINT_GEN_H = "models/genh.pth.tar"
CHECKPOINT_GEN_Z = "models/genz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        # A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ],
    additional_targets={"image0":"image"},
)

