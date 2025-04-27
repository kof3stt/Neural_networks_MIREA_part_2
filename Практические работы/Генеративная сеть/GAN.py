import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional, Callable


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('gan_textures', exist_ok=True)


class TexturesDataset(Dataset):
    """Кастомный датасет для загрузки текстурных изображений из папки."""

    def __init__(self, root_dir: str, transform: Optional[Callable] = None) -> None:
        """
        Args:
            root_dir (str): Путь к директории с изображениями.
            transform (Callable, optional): Трансформации, применяемые к изображениям.
        """
        self.paths = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', 'jpeg', 'bmp'))
        ])
        self.transform = transform

    def __len__(self) -> int:
        """Возвращает количество изображений в датасете."""
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Загружает и возвращает одно изображение по индексу."""
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


transform = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

dataset = TexturesDataset('textures', transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                        num_workers=0, pin_memory=False)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        if m.weight is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Generator(nn.Module):
    """Генератор изображений для GAN."""

    def __init__(self, latent_dim: int = 100) -> None:
        """
        Args:
            latent_dim (int): Размерность латентного вектора.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 512*4*4)

        def up(in_c: int, out_c: int) -> nn.Sequential:
            """Строит блок апсемплинга."""
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, 1),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(True),
            )

        self.net = nn.Sequential(
            self.fc,
            nn.ReLU(True),
            nn.Unflatten(1, (512, 4, 4)),
            up(512, 512),  # 4 → 8
            up(512, 256),  # 8 → 16
            up(256, 256),  # 16 → 32
            up(256, 128),  # 32 → 64
            up(128,  64),  # 64 → 128
            up(64,   32),  # 128 → 256
            up(32,   16),  # 256 → 512
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 512 → 1024
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Прямой проход генератора."""
        return self.net(z)


class Discriminator(nn.Module):
    """Дискриминатор для оценки реальности изображений."""

    def __init__(self) -> None:
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            """Строит сверточный блок."""
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            )

        self.features = nn.Sequential(
            block(3,   64),  # 1024 → 512
            block(64, 128),  # 512 → 256
            block(128, 256), # 256 → 128
            block(256, 512), # 128 → 64
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход дискриминатора."""
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


latent_dim = 128
epochs = 200
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)
G.apply(weights_init)
D.apply(weights_init)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.999))

g_losses, d_losses = [], []
real_probs, fake_probs = [], []


for epoch in range(1, epochs + 1):
    ep_g, ep_d = 0.0, 0.0
    ep_r, ep_f = 0.0, 0.0
    nb = len(dataloader)

    for i, real in enumerate(dataloader, 1):
        real = real.to(device)
        b = real.size(0)

        valid = torch.full((b,1), 0.9, device=device)
        fake_lbl = torch.full((b,1), 0.1, device=device)

        opt_D.zero_grad()
        out_r = D(real)
        loss_r = criterion(out_r, valid)

        z = torch.randn(b, latent_dim, device=device)
        fake = G(z).detach()
        out_f = D(fake)
        loss_f = criterion(out_f, fake_lbl)

        d_loss = 0.5*(loss_r + loss_f)
        d_loss.backward()
        opt_D.step()

        for _ in range(2):
            opt_G.zero_grad()
            z2 = torch.randn(b, latent_dim, device=device)
            gen = G(z2)
            out_gen = D(gen)
            g_loss = criterion(out_gen, valid)
            g_loss.backward()
            opt_G.step()

        ep_d += d_loss.item()
        ep_g += g_loss.item()
        ep_r += out_r.mean().item()
        ep_f += out_f.mean().item()

        print(f"[{epoch:03d}/{epochs}]"
              f" [{i:03d}/{nb:03d}]"
              f" D_loss:{d_loss:.4f}"
              f" G_loss:{g_loss:.4f}"
              f" R:{out_r.mean().item():.2f}"
              f" F:{out_f.mean().item():.2f}")

    d_losses.append(ep_d/nb)
    g_losses.append(ep_g/nb)
    real_probs.append(ep_r/nb)
    fake_probs.append(ep_f/nb)

    save_dir = os.path.join('gan_textures', f'epoch{epoch}')
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        samp_z = torch.randn(32, latent_dim, device=device)
        samples = G(samp_z).cpu()
        for idx, img in enumerate(samples,1):
            torchvision.utils.save_image(img,
                os.path.join(save_dir, f"{idx:02d}.png"),
                normalize=True, value_range=(-1,1)
            )

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(g_losses,'-o',label='G Loss')
    plt.plot(d_losses,'-o',label='D Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(real_probs,'-o',label='D(real)')
    plt.plot(fake_probs,'-o',label='D(fake)')
    plt.title('D Outputs')
    plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('gan_textures','metrics.png'))
    plt.close()

print("Done. Все результаты — в папке gan_textures/")  
