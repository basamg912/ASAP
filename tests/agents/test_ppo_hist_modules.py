import torch
from humanoidverse.agents.modules.ppo_hist_modules import MLP, HistoryEncoder


def test_mlp_shape():
    net = MLP(10, 4, [16, 8])
    out = net(torch.zeros(5, 10))
    assert out.shape == (5, 4)


def test_history_encoder_shapes():
    enc = HistoryEncoder(history_dim=40, latent_dim=16)
    mu, logvar = enc(torch.zeros(7, 40))
    assert mu.shape == (7, 16)
    assert logvar.shape == (7, 16)


def test_history_encoder_sample_train_is_stochastic():
    torch.manual_seed(0)
    enc = HistoryEncoder(history_dim=40, latent_dim=16)
    enc.train()
    h = torch.randn(3, 40)
    z1, _, _ = enc.sample(h)
    z2, _, _ = enc.sample(h)
    assert not torch.allclose(z1, z2)


def test_history_encoder_sample_eval_is_mean():
    enc = HistoryEncoder(history_dim=40, latent_dim=16)
    enc.eval()
    h = torch.randn(3, 40)
    z, mu, _ = enc.sample(h)
    assert torch.allclose(z, mu)


def test_state_predictor_shape():
    from humanoidverse.agents.modules.ppo_hist_modules import StatePredictor
    dec = StatePredictor(latent_dim=16, recon_dim=68)
    out = dec(torch.zeros(9, 16))
    assert out.shape == (9, 68)


def test_vae_kl_zero_at_standard_normal():
    from humanoidverse.agents.modules.ppo_hist_modules import vae_kl_loss
    mu = torch.zeros(4, 16)
    logvar = torch.zeros(4, 16)  # var = 1
    assert torch.allclose(vae_kl_loss(mu, logvar), torch.tensor(0.0), atol=1e-6)


def test_vae_kl_positive_when_offset():
    from humanoidverse.agents.modules.ppo_hist_modules import vae_kl_loss
    mu = torch.ones(4, 16)
    logvar = torch.zeros(4, 16)
    assert vae_kl_loss(mu, logvar).item() > 0.0


def test_recon_mask_all_zero_gives_zero_over_valid():
    from humanoidverse.agents.modules.ppo_hist_modules import recon_loss_masked
    pred = torch.ones(3, 5)
    target = torch.zeros(3, 5)
    mask = torch.zeros(3)          # no valid samples
    assert torch.allclose(recon_loss_masked(pred, target, mask), torch.tensor(0.0))


def test_recon_mask_selects_valid_samples():
    from humanoidverse.agents.modules.ppo_hist_modules import recon_loss_masked
    pred = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    target = torch.zeros(2, 2)
    mask = torch.tensor([1.0, 0.0])   # only first sample counts, its MSE = 1.0
    assert torch.allclose(recon_loss_masked(pred, target, mask), torch.tensor(1.0))
