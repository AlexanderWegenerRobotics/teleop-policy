import torch
import torch.nn as nn

from .backbone import ResNet18Backbone, sine_pos_embed_1d, sine_pos_embed_2d


class ACT(nn.Module):
    """CVAE-conditioned transformer that predicts a K-step action chunk."""

    def __init__(self, n_cameras, proprio_dim=20, action_dim=20, chunk_size=60,
                 hidden_dim=512, latent_dim=32, nheads=8, enc_layers=4, dec_layers=7,
                 dim_feedforward=3200, dropout=0.1):
        """Build backbones, CVAE encoder, transformer and action head."""
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.backbones = nn.ModuleList([ResNet18Backbone() for _ in range(n_cameras)])
        self.input_proj = nn.Conv2d(self.backbones[0].num_channels, hidden_dim, kernel_size=1)

        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.extra_pos_embed = nn.Embedding(2, hidden_dim)

        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.cvae_proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        self.cvae_action_proj = nn.Linear(action_dim, hidden_dim)
        cvae_layer = nn.TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout, batch_first=True)
        self.cvae_encoder = nn.TransformerEncoder(cvae_layer, enc_layers)
        self.latent_head = nn.Linear(hidden_dim, latent_dim * 2)

        enc_layer = nn.TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers)
        dec_layer = nn.TransformerDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)

        self.query_embed = nn.Embedding(chunk_size, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def _encode_images(self, images):
        """Images [B, n_cam, 3, H, W] to flattened tokens and position embeddings."""
        B, n_cam = images.shape[:2]
        tokens, poses = [], []
        for i in range(n_cam):
            feat = self.input_proj(self.backbones[i](images[:, i]))
            h, w = feat.shape[-2:]
            pos = sine_pos_embed_2d(h, w, self.hidden_dim, feat.device)
            tokens.append(feat.flatten(2).transpose(1, 2))
            poses.append(pos.flatten(1).transpose(0, 1).unsqueeze(0).expand(B, -1, -1))
        return torch.cat(tokens, dim=1), torch.cat(poses, dim=1)

    def _encode_latent(self, proprio, actions, is_pad):
        """Sample z from the CVAE encoder in training, z = 0 at inference."""
        B = proprio.shape[0]
        if actions is None:
            return proprio.new_zeros(B, self.latent_dim), None, None

        cls = self.cls_embed.weight.unsqueeze(0).expand(B, -1, -1)
        prop = self.cvae_proprio_proj(proprio).unsqueeze(1)
        acts = self.cvae_action_proj(actions)
        seq = torch.cat([cls, prop, acts], dim=1)
        pos = sine_pos_embed_1d(seq.shape[1], self.hidden_dim, proprio.device).unsqueeze(0)

        if is_pad is None:
            is_pad = torch.zeros(B, actions.shape[1], dtype=torch.bool, device=proprio.device)
        pad_mask = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=proprio.device), is_pad], dim=1)

        enc = self.cvae_encoder(seq + pos, src_key_padding_mask=pad_mask)
        mu, logvar = self.latent_head(enc[:, 0]).chunk(2, dim=-1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return z, mu, logvar

    def forward(self, images, proprio, actions=None, is_pad=None):
        """Predict an action chunk [B, K, action_dim], plus mu and logvar."""
        B = images.shape[0]
        img_tok, img_pos = self._encode_images(images)
        z, mu, logvar = self._encode_latent(proprio, actions, is_pad)

        prop_tok = self.proprio_proj(proprio).unsqueeze(1)
        lat_tok = self.latent_proj(z).unsqueeze(1)
        extra_pos = self.extra_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)

        tokens = torch.cat([img_tok, prop_tok, lat_tok], dim=1)
        pos = torch.cat([img_pos, extra_pos], dim=1)

        memory = self.encoder(tokens + pos)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        a_hat = self.action_head(self.decoder(queries, memory))
        return a_hat, mu, logvar


def kl_divergence(mu, logvar):
    """KL divergence to a standard normal, summed over latent dims, batch mean."""
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


def act_loss(a_hat, actions, is_pad, mu, logvar, kl_weight=10.0):
    """L1 on unpadded chunk steps plus weighted KL."""
    valid = (~is_pad).unsqueeze(-1)
    l1 = ((a_hat - actions).abs() * valid).sum() / valid.sum() / actions.shape[-1]
    kl = kl_divergence(mu, logvar) if mu is not None else a_hat.new_zeros(())
    return l1 + kl_weight * kl, l1, kl
