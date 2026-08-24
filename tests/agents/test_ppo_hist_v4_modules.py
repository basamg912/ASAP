import torch

from humanoidverse.agents.modules.ppo_hist_v4_modules import (
    TeacherEncoder, TeacherEncoderContrastive, TeacherValueHead,
    vicreg_var_loss, vicreg_cov_loss,
)


def test_teacher_encoder_shape_is_latent_dim_only():
    """v2 TeacherVAE 는 2*latent (mu, logvar) 였다. v4 는 latent 만 출력한다."""
    enc = TeacherEncoder(obs_dim=68, latent_dim=16)
    out = enc(torch.randn(7, 68))
    assert out.shape == (7, 16)


def test_teacher_encoder_has_no_decoder():
    enc = TeacherEncoder(obs_dim=68, latent_dim=16)
    names = [n for n, _ in enc.named_modules()]
    assert not any("decoder" in n for n in names)


def test_teacher_encoder_is_deterministic_in_train_mode():
    """VAE 와 달리 sampling 이 없으므로 train/eval 모두 같은 출력."""
    enc = TeacherEncoder(obs_dim=68, latent_dim=16)
    obs = torch.randn(5, 68)
    enc.train()
    z_train_1, z_train_2 = enc(obs), enc(obs)
    enc.eval()
    z_eval = enc(obs)
    assert torch.allclose(z_train_1, z_train_2)
    assert torch.allclose(z_train_1, z_eval)


def test_teacher_output_norm_standardizes_latent():
    enc = TeacherEncoder(obs_dim=68, latent_dim=16, output_norm=True)
    z = enc(torch.randn(64, 68))
    assert torch.allclose(z.mean(dim=-1), torch.zeros(64), atol=1e-5)
    assert torch.allclose(z.std(dim=-1, unbiased=False), torch.ones(64), atol=1e-3)


def test_teacher_output_norm_off_is_raw_mlp():
    enc = TeacherEncoder(obs_dim=68, latent_dim=16, output_norm=False)
    obs = torch.randn(4, 68)
    assert torch.allclose(enc(obs), enc.net(obs))


def test_contrastive_teacher_encoder_has_projection_head():
    enc = TeacherEncoderContrastive(
        obs_dim=68, latent_dim=16, proj_dim=12, proj_hidden_dims=[24])
    z = enc(torch.randn(7, 68))
    projected = enc.project_teacher_latent(z)
    assert z.shape == (7, 16)
    assert projected.shape == (7, 12)


def test_frozen_teacher_gets_no_gradient_from_student_loss():
    """frozen 모드 학습 경로: 타깃은 detach 되어 teacher 로 grad 가 흐르지 않는다."""
    teacher = TeacherEncoder(obs_dim=68, latent_dim=16)
    teacher.requires_grad_(False)
    student_z = torch.randn(6, 16, requires_grad=True)
    with torch.no_grad():
        z_t = teacher(torch.randn(6, 68))
    (student_z - z_t.detach()).pow(2).mean().backward()
    assert student_z.grad is not None and student_z.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())


def test_vicreg_var_loss_penalizes_collapse():
    collapsed = torch.ones(32, 16)                       # 모든 샘플 동일 -> std 0
    unit = torch.randn(4096, 16)                         # std ~ 1
    assert vicreg_var_loss(collapsed).item() > 0.9       # hinge(1 - 0) ~ 1
    assert vicreg_var_loss(unit).item() < 0.05


def test_vicreg_var_loss_is_zero_above_target_std():
    z = torch.randn(4096, 16) * 3.0
    assert torch.allclose(vicreg_var_loss(z), torch.tensor(0.0))


def test_vicreg_cov_loss_zero_for_uncorrelated_and_positive_for_duplicated():
    torch.manual_seed(0)
    indep = torch.randn(8192, 4)
    dup = indep[:, :1].repeat(1, 4)                       # 모든 차원이 같은 정보 (완전 중복)
    assert vicreg_cov_loss(indep).item() < 0.05
    assert vicreg_cov_loss(dup).item() > 0.5


def test_vicreg_losses_are_safe_on_single_sample():
    z = torch.randn(1, 16)
    assert vicreg_var_loss(z).item() == 0.0
    assert vicreg_cov_loss(z).item() == 0.0


def test_vicreg_teacher_gradient_flows():
    """vicreg 모드: align + var/cov 가 teacher 파라미터로 grad 를 흘린다."""
    teacher = TeacherEncoder(obs_dim=68, latent_dim=16)
    z_t = teacher(torch.randn(64, 68))
    student_z = torch.randn(64, 16)
    loss = ((z_t - student_z).pow(2).mean()
            + vicreg_var_loss(z_t) + 0.04 * vicreg_cov_loss(z_t))
    loss.backward()
    grad = next(teacher.net.parameters()).grad
    assert grad is not None and grad.abs().sum() > 0


def test_teacher_value_head_shape_and_input_layout():
    head = TeacherValueHead(critic_obs_dim=45, latent_dim=16, hidden_dims=[32])
    out = head(torch.randn(11, 45), torch.randn(11, 16))
    assert out.shape == (11, 1)
    assert head.net.net[0].in_features == 45 + 16


def test_teacher_value_head_gradient_reaches_teacher_encoder():
    """critic 모드 학습 경로: return 회귀 오차가 teacher encoder 까지 흘러야 한다."""
    teacher = TeacherEncoder(obs_dim=68, latent_dim=16, hidden_dims=[32])
    head = TeacherValueHead(critic_obs_dim=45, latent_dim=16, hidden_dims=[32])
    z_t = teacher(torch.randn(32, 68))
    v_aux = head(torch.randn(32, 45), z_t)
    (v_aux - torch.randn(32, 1)).pow(2).mean().backward()
    enc_grad = next(teacher.net.parameters()).grad
    head_grad = next(head.net.parameters()).grad
    assert enc_grad is not None and enc_grad.abs().sum() > 0
    assert head_grad is not None and head_grad.abs().sum() > 0


def test_teacher_value_head_uses_latent_channel():
    """z_t 채널이 실제로 출력에 영향을 줘야 한다 (gradient 가 latent 로 흐르는지)."""
    head = TeacherValueHead(critic_obs_dim=45, latent_dim=16, hidden_dims=[32])
    z = torch.randn(8, 16, requires_grad=True)
    head(torch.randn(8, 45), z).sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_v4_inference_wrapper_is_student_only_v2_path():
    from humanoidverse.agents.ppo_hist_v4.inference_wrapper import HistV4InferenceModule
    from humanoidverse.agents.ppo_hist_v2.inference_wrapper import HistV2InferenceModule
    assert HistV4InferenceModule is HistV2InferenceModule
