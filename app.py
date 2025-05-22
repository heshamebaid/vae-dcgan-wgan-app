import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from torchvision.utils import save_image
import io
import base64

app = Flask(__name__)

# === VAE Model ===
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(128 * 16 * 16, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, 128 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decoder(z)
        x = x.view(x.size(0), 128, 16, 16)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# === Generator Architecture for DCGAN/WGAN ===
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 3 * 64 * 64)

        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(1024)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, z):
        z = self.relu(self.batchnorm1(self.fc1(z)))
        z = self.relu(self.batchnorm2(self.fc2(z)))
        z = self.relu(self.batchnorm3(self.fc3(z)))
        z = self.fc4(z)
        z = z.view(-1, 3, 64, 64)
        return self.tanh(z)


# === Load models and weights ===
vae = VAE(latent_dim=128)
vae.load_state_dict(torch.load(r"D:\VAE_DCGAN_WGAN\models\vae_state.pth", map_location='cpu'))
vae.eval()

dcgan_generator = Generator()
dcgan_generator.load_state_dict(torch.load(r"D:\VAE_DCGAN_WGAN\models\DCGAN_generator_state_dict.pth", map_location='cpu'))
dcgan_generator.eval()

wgan_generator = Generator()
wgan_generator.load_state_dict(torch.load(r"D:\VAE_DCGAN_WGAN\models\WGAN_generator_weights.pth", map_location='cpu'))
wgan_generator.eval()


# === Image generation helper ===
def generate_image_from_model(model_name):
    if model_name == "vae":
        z = torch.randn(1, 128)
        with torch.no_grad():
            img_tensor = vae.decode(z).detach().cpu()
        img_tensor = (img_tensor + 1) / 2
        img_tensor = img_tensor.clamp(0, 1)
        img_tensor = img_tensor.repeat(1, 3, 1, 1).squeeze(0)  # Convert to 3 channels

    elif model_name == "dcgan":
        noise = torch.randn(1, 100)
        with torch.no_grad():
            img_tensor = dcgan_generator(noise).detach().cpu().squeeze(0)
        img_tensor = (img_tensor + 1) / 2
        img_tensor = img_tensor.clamp(0, 1)

    elif model_name == "wgan":
        noise = torch.randn(1, 100)
        with torch.no_grad():
            img_tensor = wgan_generator(noise).detach().cpu().squeeze(0)
        img_tensor = (img_tensor + 1) / 2
        img_tensor = img_tensor.clamp(0, 1)

    else:
        raise ValueError("Invalid model name")

    # Convert image to base64
    buffer = io.BytesIO()
    save_image(img_tensor, buffer, format="PNG")
    buffer.seek(0)
    img_bytes = buffer.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_b64


# === Flask Routes ===
@app.route('/')
def index():
    return render_template('index.html')  # Your frontend file


@app.route('/generate', methods=['POST'])
def generate():
    model_name = request.json.get('model')
    try:
        img_b64 = generate_image_from_model(model_name)
        return jsonify({'status': 'success', 'image': img_b64})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
