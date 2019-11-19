import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.utils import save_image, make_grid

import numpy as np
import os
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

generator = Generator(input_size=(100,), img_shape=(1,28,28))
discriminator = Discriminator(784)
d_loss_fn = nn.BCELoss()
g_loss_fn = nn.BCELoss()

generator.to(device)
discriminator.to(device)
d_loss_fn.to(device)
g_loss_fn.to(device)

print('====================== Generator Model ======================')
print(summary(generator, input_size=(100,)),'\n')
print('====================== Discriminator Model ======================')
print(summary(discriminator, input_size=(784,)),'\n')

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# Loss record
epoch_train_d_losses = []
epoch_train_g_losses = []

# Fixed input for inspection of the change of generated images after training generator
sample_z = torch.randn(size=(20,100)).to(device)

for epoch in range(train_epochs):

    # Enter Train mode
    generator.train()
    discriminator.train()

    epoch_mean_d_loss = 0
    epoch_mean_g_loss = 0
    epoch_d_update_count = 0
    epoch_g_update_count = 0

    for i, (imgs, _) in enumerate(tqdm_notebook(dataloader, desc='Epoch {}/{}'.format(epoch,train_epochs))):

        # Prepare data
        imgs = imgs.reshape(batch_size,784)
        imgs = imgs + torch.tensor(torch.randn(imgs.size()) * 0.02)  # Add noise to imgs
        imgs = imgs.to(device)

        # Prepare label for BCE loss
        label_ones = torch.ones((batch_size,1)).to(device)
        label_zeros = torch.zeros((batch_size,1)).to(device)

        # Create noise samples
        z = torch.randn(size=(batch_size,100)).to(device)

        gen_imgs = generator(z).detach().reshape(batch_size,784)
        D_gen = discriminator(gen_imgs)  # Detach so no grad will flow through to generator
        D_real = discriminator(imgs)

        # ---------------------
        #  Update Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        d_real_loss = d_loss_fn(D_real,label_ones)
        d_gen_loss = d_loss_fn(D_gen, label_zeros)
        d_loss = d_real_loss + d_gen_loss
        d_loss.backward()
        optimizer_D.step()
        epoch_d_update_count += 1
        epoch_mean_d_loss += d_loss

        # -----------------
        #  Update Generator
        # -----------------
        for _ in range(1):
            optimizer_G.zero_grad()
            z = torch.randn(size=(batch_size,100)).to(device)
            gen_imgs = generator(z).reshape(batch_size,784)
            D_gen = discriminator(gen_imgs)
            g_loss = g_loss_fn(D_gen, label_ones)
            g_loss.backward()
            optimizer_G.step()
            epoch_g_update_count += 1
            epoch_mean_g_loss += g_loss

    if epoch%sample_interval == 0:
        # Enter eval mode
        generator.eval()
        discriminator.eval()

        imgs_to_save = generator(sample_z)
        imgs_to_save = make_grid(imgs_to_save, nrow=4)
        save_image(imgs_to_save, "/content/drive/My Drive/GAN_MNIST_OUTPUTS/epoch_{}.png".format(epoch))

    epoch_mean_d_loss /= epoch_d_update_count
    epoch_mean_g_loss /= epoch_g_update_count
    epoch_train_d_losses.append(epoch_mean_d_loss)
    epoch_train_g_losses.append(epoch_mean_g_loss)

    print('Epoch {}: train_d_loss = {} and train_g_loss = {}'.format(epoch, epoch_mean_d_loss, epoch_mean_g_loss))

# -----------------
#  Plot losses
# -----------------
fig = plt.figure()
plt.title('Train_D_loss and Train_G_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epoch_train_d_losses, 'r', label='Train D Loss')
plt.plot(epoch_train_g_losses, 'g', label='Train G Loss')
plt.legend(loc='upper right')
plt.show()
