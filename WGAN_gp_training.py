import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.WGAN_gp import Critic, Generator, initialize_weights, gradient_penalty

#### training images taken from https://www.kaggle.com/datasets/504743cb487a5aed565ce14238c6343b7d650ffd28c071f03f2fd9b25819e6c9?resource=download-directory

##hyperparameters
device = torch.device("mps")
lr = 5e-5
batch_size = 64
image_size = 64
CHANNELS_IMG = 3
Z_DIM = 200
NUM_EPOCHS = 4
FEATURES_critic = 64
FEATURES_GEN = 64
CRITIC_ITERATION = 5
LAMBDA = 10


transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)],
        )
    ]
)

# dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
dataset = datasets.ImageFolder(root='dataset/celeb_dataset', transform=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_critic).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))


fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(
    f"logs/real"
)
writer_fake = SummaryWriter(
    f"logs/fake"
)
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        
        for _ in range(CRITIC_ITERATION):
            noise = torch.randn((batch_size, Z_DIM, 1, 1)).to(device)
            critic_real = critic(real).reshape(-1)
            fake = gen(noise)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            lossC = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA*gp
            critic.zero_grad()
            lossC.backward(retain_graph=True)
            opt_critic.step()
        
        output = critic(fake).reshape(-1)
        lossG = -torch.mean(output)
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] \\ "
                f"loss C: {lossC:.4f}, loss G: {lossG:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise)
                # data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image(
                    "Fake Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "Real Images", img_grid_real, global_step=step
                )
                step += 1