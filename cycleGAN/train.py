import torch
from utils import save_checkpoint, save_some_examples, load_checkpoint
import torch.nn as nn
import torch.optim as optim
import config
from dataset import HorseZebraDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

def train_fn(disc_H, disc_Z, gen_H, gen_Z, loader, opt_disc, opt_gen, L1, mse):
    loop = tqdm(loader, leave=True)
    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        fake_horse = gen_H(zebra)
        D_H_real = disc_H(horse)
        D_H_fake = disc_H(fake_horse.detach())
        D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
        D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
        D_H_loss = (D_H_real_loss + D_H_fake_loss)/2

        fake_zebra = gen_Z(horse)
        D_Z_real = disc_Z(zebra)
        D_Z_fake = disc_Z(fake_zebra.detach())
        D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
        D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
        D_Z_loss = (D_Z_fake_loss + D_Z_real_loss)/2

        D_loss = D_H_loss + D_Z_loss

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        D_H_fake = disc_H(fake_horse)
        D_Z_fake = disc_Z(fake_zebra)
        loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))
        loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))

        l1 = L1
        #cycle loss
        cycle_zebra = gen_Z(fake_horse)
        cycle_horse = gen_H(fake_zebra)
        cycle_zebra_loss = l1(zebra, cycle_zebra)
        cycle_horse_loss = l1(horse, cycle_horse)

        #identity loss
        identity_zebra = gen_Z(zebra)
        identity_horse = gen_H(horse)
        identity_zebra_loss = l1(zebra, identity_zebra)
        identity_horse_loss = l1(horse, identity_horse)
        
        
        # add altogether
        G_loss = (
            loss_G_Z
            +loss_G_H
            +cycle_zebra_loss * config.LAMBDA_CYCLE
            +cycle_horse_loss * config.LAMBDA_CYCLE
            +identity_horse_loss * config.LAMBDA_IDENTITY
            +identity_zebra_loss * config.LAMBDA_IDENTITY
        )
        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        if idx % 200 == 0:
            save_image(fake_horse*0.5+0.5, f'eval/horse_{idx}.png')
            save_image(fake_zebra*0.5+0.5, f'eval/zebra_{idx}.png')

def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_DISC_H, disc_H, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_Z, disc_Z, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE)
    
    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR+'horses/', root_zebra=config.TRAIN_DIR+'zebras/', transform=config.transforms
    )
    print(config.TRAIN_DIR+'horses/', config.TRAIN_DIR+'zebras/')
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_H, disc_Z, gen_H, gen_Z, loader, opt_disc, opt_gen, L1, mse)

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_DISC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_DISC_Z)

if __name__ == '__main__':
    main()