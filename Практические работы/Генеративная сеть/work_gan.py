import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Датасет CS:GO текстур
class TexturesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png','.jpg','jpeg','bmp'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# Трансформации: 1024×1024 и нормализация
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

dataset = TexturesDataset('textures', transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                        num_workers=0, pin_memory=False)

# Инициализация весов (DCGAN-рецепт)
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

# Генератор
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512*4*4),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 4, 4)),
            *self._upsample_block(512, 512),
            *self._upsample_block(512, 256),
            *self._upsample_block(256, 256),
            *self._upsample_block(256, 128),
            *self._upsample_block(128, 64),
            *self._upsample_block(64, 32),
            *self._upsample_block(32, 16),
            nn.Upsample(scale_factor=2),  # 512->1024
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh()
        )

    def _upsample_block(self, in_ch, out_ch):
        return [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        ]

    def forward(self, z):
        return self.net(z)

# Дискриминатор
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 1024->512
            nn.LeakyReLU(0.2, True), nn.Dropout(0.25),
            nn.Conv2d(64, 128, 4, 2, 1), # 512->256
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True), nn.Dropout(0.25),
            nn.Conv2d(128, 256, 4, 2, 1),# 256->128
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True), nn.Dropout(0.25),
            nn.Conv2d(256, 512, 4, 2, 1),# 128->64
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True), nn.Dropout(0.25)
        )
        # После features: 512×64×64
        self.pool = nn.AdaptiveAvgPool2d((1,1))  # ->512×1×1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

# Гиперпараметры
latent_dim = 512
epochs = 1000

# Модели и инициализация
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)
G.apply(weights_init)
D.apply(weights_init)

# Логирование
g_losses    = []
d_losses    = []
real_probs  = []
fake_probs  = []

# Критерий и оптимизаторы
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5,0.999))
opt_D = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5,0.999))

# Папка для результатов
os.makedirs('gan_textures', exist_ok=True)

# === Тренировка ===
for epoch in range(1, epochs + 1):
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    epoch_real_p = 0.0
    epoch_fake_p = 0.0
    num_batches = len(dataloader)

    for i, real in enumerate(dataloader, 1):
        real = real.to(device)
        b = real.size(0)
        valid = torch.ones(b,1,device=device)
        fake_label = torch.zeros(b,1,device=device)

        # --- Train Discriminator ---
        opt_D.zero_grad()
        out_real = D(real)
        loss_real = criterion(out_real, valid)
        z = torch.randn(b, latent_dim, device=device)
        fake_imgs = G(z).detach()
        out_fake = D(fake_imgs)
        loss_fake = criterion(out_fake, fake_label)
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        opt_D.step()

        # --- Train Generator ---
        opt_G.zero_grad()
        out_gen = D(G(z))
        loss_G = criterion(out_gen, valid)
        loss_G.backward()
        opt_G.step()

        # Accumulate
        epoch_d_loss += loss_D.item()
        epoch_g_loss += loss_G.item()
        epoch_real_p += out_real.mean().item()
        epoch_fake_p += out_fake.mean().item()

        print(f"Epoch [{epoch}/{epochs}] Batch {i}/{num_batches} "
                f"D_loss: {loss_D:.4f} G_loss: {loss_G:.4f} "
                f"Real D: {out_real.mean().item():.2f} "
                f"Fake D: {out_fake.mean().item():.2f}")

    # Средние по эпохе
    avg_d = epoch_d_loss / num_batches
    avg_g = epoch_g_loss / num_batches
    avg_r = epoch_real_p / num_batches
    avg_f = epoch_fake_p / num_batches

    d_losses.append(avg_d)
    g_losses.append(avg_g)
    real_probs.append(avg_r)
    fake_probs.append(avg_f)

    # Сохранить 32 сэмплов (случайных) в папку epoch{N}
    save_dir = os.path.join('gan_textures', f'epoch{epoch}')
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        sample_z = torch.randn(32, latent_dim, device=device)
        samples = G(sample_z).cpu()
        for idx, img in enumerate(samples, 1):
            torchvision.utils.save_image(
                img,
                os.path.join(save_dir, f"{idx:02d}.png"),
                normalize=True
            )
    print(f"Saved 32 samples to {save_dir}")

    # Построить и сохранить графики
    plt.figure(figsize=(12, 5))
    # Losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch+1), g_losses, label='Generator')
    plt.plot(range(1, epoch+1), d_losses, label='Discriminator')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Average Loss per Epoch')
    plt.legend()
    plt.grid(True)
    # Probabilities
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch+1), real_probs, label='Real D(x)')
    plt.plot(range(1, epoch+1), fake_probs, label='D(G(z))')
    plt.xlabel('Epoch'); plt.ylabel('Probability')
    plt.title('Average D Outputs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join('gan_textures', 'training_metrics.png'))
    plt.close()

print("Training complete. All samples and metrics are in ./gan_textures/")
