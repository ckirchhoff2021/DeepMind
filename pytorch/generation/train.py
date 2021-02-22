import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms, datasets
from dcgan import Generator, Discriminator

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def to_image(x):
    out = (x + 1) * 0.5
    out = out.clamp(0, 1)
    return out


def generator_train():
    image_transform = transforms.Compose([
        transforms.Resize((ndf, ndf)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(DATA_ROOT, transform=image_transform)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=128, num_workers=2)

    D = Discriminator()
    D = D.to(DEVICE)
    D.train()

    G = Generator()
    G = G.to(DEVICE)
    G.train()

    d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)

    d_losses = list()
    g_losses = list()

    criterion = nn.BCELoss()
    for epoch in range(num_epochs):
        for index, (data, label) in enumerate(data_loader):
            number = data.size(0)
            real_data = data.to(DEVICE)
            real_label = torch.ones(number, 1).to(DEVICE)
            real_out = D(real_data)
            real_score = real_out.mean().item()
            real_loss = criterion(real_out, real_label)

            fake_x = torch.randn(number, nz, 1, 1).to(DEVICE)
            fake_data = G(fake_x)
            fake_label = torch.zeros(number, 1).to(DEVICE)
            fake_out = D(fake_data)
            fake_score = fake_out.mean().item()
            fake_loss = criterion(fake_out, fake_label)

            d_loss = real_loss + fake_loss
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            d_losses.append(d_loss.item())

            fake_x = torch.randn(64, nz, 1, 1).to(DEVICE)
            real_label = torch.ones(64, 1).to(DEVICE)
            fake_data = G(fake_x)
            fake_out = D(fake_data)
            g_loss = criterion(fake_out, real_label)

            g_losses.append(g_loss.item())

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (index + 1) % 50 == 0:
                print('Epoch: %d, d_loss: %f, g_loss: %f, real_score: %f, fake_score: %f' % (
                    epoch, d_loss.item(), g_loss.item(), real_score, fake_score
                ))

            if (index + 1) % 500 == 0:
                real_name = 'R-' + str(epoch) + '_' + str(index + 1) + '.png'
                real_image = to_image(real_data)
                save_image(real_image, os.path.join(IMAGE_SAVE_PATH, real_name))

                fake_name = 'F-' + str(epoch) + '_' + str(index + 1) + '.png'
                fake_image = to_image(fake_data)
                save_image(fake_image, os.path.join(IMAGE_SAVE_PATH, fake_name))

            torch.save(D, os.path.join(MODEL_SAVE_PATH, 'discriminator.pkl'))
            torch.save(G, os.path.join(MODEL_SAVE_PATH, 'generator.pkl'))

    with open(os.path.join(MODEL_SAVE_PATH, 'd_loss.json'), 'w') as f:
        json.dump(d_losses, f)

    with open(os.path.join(MODEL_SAVE_PATH, 'g_loss.json'), 'w') as f:
        json.dump(g_losses, f)


def main():
    generator_train()


if __name__ == '__main__':
    generator_train()