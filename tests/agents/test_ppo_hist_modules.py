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
