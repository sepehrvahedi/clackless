"""Export DeepFilterNet3 as a single-frame streaming ONNX model.

Produces ``src/clackless/models/deepfilternet3_streaming.onnx``: one call takes
10 ms of 48 kHz audio (480 samples) plus the model's recurrent state and returns
10 ms of enhanced audio plus the new state. Everything (STFT, ERB features,
normalisation, the three networks, deep filtering, overlap-add) is inside the
graph, so the runtime needs only onnxruntime and NumPy.

The streaming formulation is adapted from the TorchDF port by grazder
(https://github.com/grazder/DeepFilterNet, branch torchDF_main), itself derived
from DeepFilterNet by Hendrik Schröter (https://github.com/Rikorose/DeepFilterNet),
both MIT / Apache-2.0. The weights are the official DeepFilterNet3 release.

Needs a separate environment (not a runtime dependency of Clackless):

    pip install deepfilternet==0.5.6 torch==2.5.1 torchaudio==2.5.1 onnx onnxruntime soundfile
    python scripts/export_deepfilternet.py --check some_noisy_speech_48k.wav
"""

from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401  (kept for parity with df.modules)
from df import init_df
from df.config import Csv, DfParams, config
from df.modules import Conv2dNormAct, ConvTranspose2dNormAct
from torch import Tensor, nn
from torch.nn import init
from torch.nn.parameter import Parameter

OUT = Path(__file__).resolve().parents[1] / "src" / "clackless" / "models" / "deepfilternet3_streaming.onnx"


class GroupedLinearEinsum(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, groups: int = 1, activation: Optional[nn.Module] = None
    ):
        super().__init__()
        self.input_size, self.hidden_size, self.groups = input_size, hidden_size, groups
        self.ws = input_size // groups
        self.register_parameter(
            "weight", Parameter(torch.zeros(groups, input_size // groups, hidden_size // groups))
        )
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(self.groups, 1, self.ws)
        x = torch.matmul(x, self.weight)
        x = self.activation(x)
        return x.view(1, 1, -1)


class SqueezedGRU_S(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
        gru_skip_op: Optional[Callable[..., nn.Module]] = None,
        linear_act_layer: Callable[..., nn.Module] = nn.Identity,
    ):
        super().__init__()
        self.linear_in = nn.Sequential(
            GroupedLinearEinsum(input_size, hidden_size, linear_groups, linear_act_layer())
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinearEinsum(hidden_size, output_size, linear_groups, linear_act_layer())
            )
        else:
            self.linear_out = nn.Identity()

    def forward(self, input: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.linear_in(input)
        x, h = self.gru(x, h)
        x = self.linear_out(x)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        return x, h


class ModelParams(DfParams):
    section = "deepfilternet"

    def __init__(self):
        super().__init__()
        sec = self.section
        self.conv_lookahead: int = config("CONV_LOOKAHEAD", cast=int, default=0, section=sec)
        self.conv_ch: int = config("CONV_CH", cast=int, default=16, section=sec)
        self.conv_kernel: List[int] = config("CONV_KERNEL", cast=Csv(int), default=(1, 3), section=sec)
        self.convt_kernel: List[int] = config("CONVT_KERNEL", cast=Csv(int), default=(1, 3), section=sec)
        self.conv_kernel_inp: List[int] = config(
            "CONV_KERNEL_INP", cast=Csv(int), default=(3, 3), section=sec
        )
        self.emb_hidden_dim: int = config("EMB_HIDDEN_DIM", cast=int, default=256, section=sec)
        self.emb_num_layers: int = config("EMB_NUM_LAYERS", cast=int, default=2, section=sec)
        self.emb_gru_skip_enc: str = config("EMB_GRU_SKIP_ENC", default="none", section=sec)
        self.emb_gru_skip: str = config("EMB_GRU_SKIP", default="none", section=sec)
        self.df_hidden_dim: int = config("DF_HIDDEN_DIM", cast=int, default=256, section=sec)
        self.df_gru_skip: str = config("DF_GRU_SKIP", default="none", section=sec)
        self.df_pathway_kernel_size_t: int = config(
            "DF_PATHWAY_KERNEL_SIZE_T", cast=int, default=1, section=sec
        )
        self.df_num_layers: int = config("DF_NUM_LAYERS", cast=int, default=3, section=sec)
        self.lin_groups: int = config("LINEAR_GROUPS", cast=int, default=1, section=sec)
        self.enc_lin_groups: int = config("ENC_LINEAR_GROUPS", cast=int, default=16, section=sec)


def _skip(kind: str, in_dim: int, out_dim: int, groups: int):
    if kind == "none":
        return None
    if kind == "identity":
        return partial(nn.Identity)
    if kind == "groupedlinear":
        return partial(GroupedLinearEinsum, input_size=in_dim, hidden_size=out_dim, groups=groups)
    raise NotImplementedError(kind)


def _unpad(conv: nn.Module) -> nn.Module:
    """The streaming model feeds the convolutions a rolling buffer instead of padding."""
    return nn.Sequential(*[m for m in conv if not isinstance(m, nn.ConstantPad2d)])


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        conv = partial(
            Conv2dNormAct,
            in_ch=p.conv_ch,
            out_ch=p.conv_ch,
            kernel_size=p.conv_kernel,
            bias=False,
            separable=True,
        )
        self.erb_conv0 = Conv2dNormAct(
            1, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False, separable=True
        )
        self.erb_conv1 = conv(fstride=2)
        self.erb_conv2 = conv(fstride=2)
        self.erb_conv3 = conv(fstride=1)
        self.df_conv0 = Conv2dNormAct(2, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False, separable=True)
        self.df_conv1 = conv(fstride=2)
        self.df_conv0_ch = p.conv_ch
        emb_in = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim
        self.df_fc_emb = nn.Sequential(
            GroupedLinearEinsum(
                p.conv_ch * p.nb_df // 2, emb_in, groups=p.enc_lin_groups, activation=nn.ReLU(inplace=True)
            )
        )
        self.emb_gru = SqueezedGRU_S(
            emb_in,
            self.emb_dim,
            output_size=emb_in,
            num_layers=1,
            batch_first=False,
            gru_skip_op=_skip(p.emb_gru_skip_enc, emb_in, emb_in, p.lin_groups),
            linear_groups=p.lin_groups,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        self.lsnr_fc = nn.Sequential(nn.Linear(emb_in, 1), nn.Sigmoid())

    def forward(self, feat_erb: Tensor, feat_spec: Tensor, hidden: Tensor):
        e0 = self.erb_conv0(feat_erb)
        e1 = self.erb_conv1(e0)
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)
        emb = e3.permute(0, 2, 3, 1).flatten(2, 3)
        c0 = self.df_conv0(feat_spec)
        c1 = self.df_conv1(c0)
        cemb = self.df_fc_emb(c1.permute(0, 2, 3, 1))
        emb, hidden = self.emb_gru(emb + cemb, hidden)
        return e0, e1, e2, e3, emb, c0, hidden


class ErbDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        emb_in = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim
        self.emb_gru = SqueezedGRU_S(
            emb_in,
            self.emb_dim,
            output_size=emb_in,
            num_layers=p.emb_num_layers - 1,
            batch_first=False,
            gru_skip_op=_skip(p.emb_gru_skip, emb_in, emb_in, p.lin_groups),
            linear_groups=p.lin_groups,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        tconv = partial(ConvTranspose2dNormAct, kernel_size=p.convt_kernel, bias=False, separable=True)
        conv = partial(Conv2dNormAct, bias=False, separable=True)
        self.conv3p = conv(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt3 = conv(p.conv_ch, p.conv_ch, kernel_size=p.conv_kernel)
        self.conv2p = conv(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt2 = tconv(p.conv_ch, p.conv_ch, fstride=2)
        self.conv1p = conv(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt1 = tconv(p.conv_ch, p.conv_ch, fstride=2)
        self.conv0p = conv(p.conv_ch, p.conv_ch, kernel_size=1)
        self.conv0_out = conv(p.conv_ch, 1, kernel_size=p.conv_kernel, activation_layer=nn.Sigmoid)

    def forward(self, emb, e3, e2, e1, e0, hidden):
        f8 = e3.shape[-1]
        emb, hidden = self.emb_gru(emb, hidden)
        emb = emb.view(1, 1, f8, -1).permute(0, 3, 1, 2)
        e3 = self.convt3(self.conv3p(e3) + emb)
        e2 = self.convt2(self.conv2p(e2) + e3)
        e1 = self.convt1(self.conv1p(e1) + e2)
        return self.conv0_out(self.conv0p(e0) + e1), hidden


class DfDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        emb_in = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.df_hidden_dim
        self.df_bins = p.nb_df
        self.df_out_ch = p.df_order * 2
        self.df_convp = Conv2dNormAct(
            p.conv_ch,
            self.df_out_ch,
            fstride=1,
            kernel_size=(p.df_pathway_kernel_size_t, 1),
            separable=True,
            bias=False,
        )
        self.df_gru = SqueezedGRU_S(
            emb_in,
            self.emb_dim,
            num_layers=p.df_num_layers,
            batch_first=True,
            gru_skip_op=None,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        kind = p.df_gru_skip.lower()
        self.df_skip = None
        if kind == "identity":
            self.df_skip = nn.Identity()
        elif kind == "groupedlinear":
            self.df_skip = GroupedLinearEinsum(emb_in, self.emb_dim, groups=p.lin_groups)
        self.df_out = nn.Sequential(
            GroupedLinearEinsum(
                p.df_hidden_dim, p.nb_df * self.df_out_ch, groups=p.lin_groups, activation=nn.Tanh()
            )
        )
        self.df_fc_a = nn.Sequential(nn.Linear(p.df_hidden_dim, 1), nn.Sigmoid())

    def forward(self, emb, c0, hidden):
        b, t, _ = emb.shape
        c, hidden = self.df_gru(emb, hidden)
        if self.df_skip is not None:
            c = c + self.df_skip(emb)
        c0 = self.df_convp(c0).permute(0, 2, 3, 1)
        c = self.df_out(c).view(b, t, self.df_bins, self.df_out_ch) + c0
        return c, hidden


class StreamingDF(nn.Module):
    """One 10 ms frame in, one 10 ms frame out; all state passed explicitly."""

    STATE_NAMES = [
        "erb_norm",
        "unit_norm",
        "analysis_mem",
        "synthesis_mem",
        "erb_buf",
        "spec_feat_buf",
        "c0_buf",
        "spec_buf_x",
        "spec_buf_y",
        "enc_hidden",
        "erb_dec_hidden",
        "df_dec_hidden",
    ]

    def __init__(self, model, erb_widths: np.ndarray, p: ModelParams, alpha: float):
        super().__init__()
        self.hop = p.hop_size
        self.n_fft = p.fft_size
        self.n_freq = p.fft_size // 2 + 1
        self.nb_erb, self.nb_df, self.df_order, self.alpha = p.nb_erb, p.nb_df, p.df_order, alpha
        self.wnorm = 1.0 / (self.n_fft**2 / (2 * self.hop))

        # Vorbis window, as in libDF.
        n = torch.arange(self.n_fft, dtype=torch.float64)
        w = torch.sin(0.5 * torch.pi * (n + 0.5) / (self.n_fft // 2))
        self.register_buffer("window", torch.sin(0.5 * torch.pi * w**2).float())
        # Real DFT / inverse as matrices: plain MatMul in ONNX, no FFT operator needed.
        rfft = torch.view_as_real(torch.fft.rfft(torch.eye(self.n_fft, dtype=torch.float64))).transpose(0, 1)
        self.register_buffer("rfft_matrix", rfft.float())  # [F, N, 2]
        self.register_buffer("irfft_matrix", torch.linalg.pinv(rfft).float())  # [F, 2, N]

        widths = torch.from_numpy(erb_widths.astype(np.int64))
        fb = torch.zeros(int(widths.sum()), widths.numel())
        start = 0
        for i, wd in enumerate(widths.tolist()):
            fb[start : start + wd, i] = 1
            start += wd
        self.register_buffer("erb_fwd", fb / fb.sum(dim=0))
        self.register_buffer("erb_inv", fb.t().contiguous())

        self.enc, self.erb_dec, self.df_dec = Encoder(), ErbDecoder(), DfDecoder()
        self.enc.load_state_dict(model.enc.state_dict())
        self.erb_dec.load_state_dict(model.erb_dec.state_dict())
        self.df_dec.load_state_dict(model.df_dec.state_dict())
        self.enc.erb_conv0 = _unpad(self.enc.erb_conv0)
        self.enc.df_conv0 = _unpad(self.enc.df_conv0)
        self.df_dec.df_convp = _unpad(self.df_dec.df_convp)
        self.eval()

        la = p.conv_lookahead
        self.initial_states = [
            torch.linspace(-60.0, -90.0, self.nb_erb),
            torch.linspace(0.001, 0.0001, self.nb_df).view(1, self.nb_df, 1),
            torch.zeros(self.hop),
            torch.zeros(self.n_fft - self.hop),
            torch.zeros(1, 1, la + 1, self.nb_erb),
            torch.zeros(1, 2, la + 1, self.nb_df),
            torch.zeros(1, self.enc.df_conv0_ch, self.df_order, self.nb_df),
            torch.zeros(max(self.df_order, la), self.n_freq, 2),
            torch.zeros(self.df_order + la, self.n_freq, 2),
            torch.zeros(1, 1, self.enc.emb_dim),
            torch.zeros(p.emb_num_layers - 1, 1, self.erb_dec.emb_dim),
            torch.zeros(p.df_num_layers, 1, self.df_dec.emb_dim),
        ]

    def forward(
        self,
        frame,
        erb_norm,
        unit_norm,
        analysis_mem,
        synthesis_mem,
        erb_buf,
        spec_feat_buf,
        c0_buf,
        spec_buf_x,
        spec_buf_y,
        enc_hidden,
        erb_dec_hidden,
        df_dec_hidden,
    ):
        # Analysis: [previous 10 ms, this 10 ms] * window -> spectrum [F, 2].
        buf = torch.cat([analysis_mem, frame]) * self.window
        spec = torch.matmul(buf, self.rfft_matrix) * self.wnorm
        spec = spec.unsqueeze(0)
        spec_buf_x = torch.cat([spec_buf_x[1:], spec])
        spec_buf_y = torch.cat([spec_buf_y[1:], spec])

        # Features: ERB levels (dB) and the complex low band, both running-normalised.
        erb_db = 10.0 * torch.log10(torch.sum(spec**2, dim=-1).matmul(self.erb_fwd) + 1e-10).squeeze(0)
        erb_norm = erb_db * (1 - self.alpha) + erb_norm * self.alpha
        erb_feat = ((erb_db - erb_norm) / 40.0)[None, None, None, :]
        low = spec[:, : self.nb_df]
        unit_norm = torch.linalg.norm(low, dim=-1, keepdim=True) * (1 - self.alpha) + unit_norm * self.alpha
        spec_feat = (low / unit_norm.sqrt())[None].permute(0, 3, 1, 2)

        erb_buf = torch.cat([erb_buf[:, :, 1:], erb_feat], dim=2)
        spec_feat_buf = torch.cat([spec_feat_buf[:, :, 1:], spec_feat], dim=2)
        e0, e1, e2, e3, emb, c0, enc_hidden = self.enc(erb_buf, spec_feat_buf, enc_hidden)
        gains, erb_dec_hidden = self.erb_dec(emb, e3, e2, e1, e0, erb_dec_hidden)
        c0_buf = torch.cat([c0_buf[:, :, 1:], c0], dim=2)
        coefs, df_dec_hidden = self.df_dec(emb, c0_buf, df_dec_hidden)
        coefs = coefs.view(self.nb_df, -1, 2).permute(1, 0, 2)  # [order, DF, 2]

        # Enhance the frame `conv_lookahead` frames back: ERB gains on all bins,
        # then a complex deep filter over the last `df_order` frames on the low band.
        cur = spec_buf_y[self.df_order - 1] * gains.view(self.nb_erb).matmul(self.erb_inv).unsqueeze(-1)
        xs = spec_buf_x[:, : self.nb_df]
        re = xs[..., 0] * coefs[..., 0] - xs[..., 1] * coefs[..., 1]
        im = xs[..., 0] * coefs[..., 1] + xs[..., 1] * coefs[..., 0]
        low_out = torch.stack([re.sum(0), im.sum(0)], dim=-1)
        cur = torch.cat([low_out, cur[self.nb_df :]], dim=0)

        # Synthesis with overlap-add.
        y = torch.einsum("fi,fij->j", cur, self.irfft_matrix) * float(self.n_fft) * self.window
        out = y[: self.hop] + synthesis_mem[: self.hop]
        synthesis_mem = torch.cat([synthesis_mem[self.hop :], torch.zeros(self.hop)]) + y[self.hop :]
        return (
            out,
            erb_norm,
            unit_norm,
            frame,
            synthesis_mem,
            erb_buf,
            spec_feat_buf,
            c0_buf,
            spec_buf_x,
            spec_buf_y,
            enc_hidden,
            erb_dec_hidden,
            df_dec_hidden,
        )


def build() -> StreamingDF:
    model, state, _ = init_df(config_allow_defaults=True, model_base_dir="DeepFilterNet3", epoch="best")
    model.eval()
    p = ModelParams()
    alpha = round(float(np.exp(-p.hop_size / p.sr / p.norm_tau)), 3)  # libDF rounds it the same way
    return StreamingDF(model, state.erb_widths(), p, alpha)


def run_stream(step, states, audio: np.ndarray, hop: int) -> np.ndarray:
    pad = (-audio.size) % hop
    x = np.concatenate([audio, np.zeros(pad)]).astype(np.float32)
    out = []
    for i in range(0, x.size, hop):
        res = step(x[i : i + hop], states)
        out.append(res[0])
        states = res[1:]
    return np.concatenate(out)[: audio.size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", type=Path, help="48 kHz wav: compare against the offline df.enhance")
    args = ap.parse_args()
    torch.set_num_threads(1)
    m = build()
    names_in = ["frame"] + m.STATE_NAMES
    names_out = ["out"] + ["new_" + n for n in m.STATE_NAMES]
    example = (torch.zeros(m.hop), *m.initial_states)
    with torch.no_grad():
        torch.onnx.export(
            m,
            example,
            str(args.out),
            input_names=names_in,
            output_names=names_out,
            opset_version=17,
            dynamo=False,
        )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), alpha={m.alpha:.6f}")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    init_np = [s.numpy() for s in m.initial_states]

    def ort_step(frame, states):
        return sess.run(None, {"frame": frame, **dict(zip(m.STATE_NAMES, states, strict=True))})

    def torch_step(frame, states):
        with torch.no_grad():
            r = m(torch.from_numpy(frame), *[torch.from_numpy(np.asarray(s)) for s in states])
        return [t.numpy() for t in r]

    rng = np.random.default_rng(0)
    noise = rng.standard_normal(48000).astype(np.float32) * 0.1
    a = run_stream(torch_step, init_np, noise, m.hop)
    b = run_stream(ort_step, init_np, noise, m.hop)
    print("torch vs onnx max abs diff:", float(np.max(np.abs(a - b))))

    if args.check:
        import soundfile as sf
        from df.enhance import enhance

        audio, sr = sf.read(args.check, dtype="float32")
        assert sr == 48000
        audio = audio if audio.ndim == 1 else audio.mean(axis=1)
        model, df_state, _ = init_df(
            config_allow_defaults=True, model_base_dir="DeepFilterNet3", epoch="best"
        )
        ref = enhance(model, df_state, torch.from_numpy(audio)[None]).numpy()[0]
        got = run_stream(ort_step, init_np, audio, m.hop)
        # Find the streaming delay, then compare.
        lags = range(0, 4 * m.hop + 1, 1)
        best = max(lags, key=lambda d: float(np.dot(got[d:], ref[: ref.size - d])))
        g, r = got[best:], ref[: ref.size - best]
        snr = 10 * np.log10(np.sum(r**2) / (np.sum((g - r) ** 2) + 1e-12))
        print(
            f"streaming delay {best} samples ({best / 48:.1f} ms); agreement with offline: {snr:.1f} dB SNR"
        )


if __name__ == "__main__":
    main()
