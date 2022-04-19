# %%
import torch
import torchaudio
import math
from torchaudio.compliance import kaldi
# %%


def istft(signal, nfft=2046, nhop=64):
    signal = torch.from_numpy(signal).permute(1, 2, 0)
    #mag = torch.norm(signal,2,2,True)
    t_recon = torchaudio.functional.istft(signal, nfft, nhop)
    #t_recon = torch.zeros_like(t).uniform_(-1, 1)
    #for x in range(100):
    #    spec_recon = torch.stft(t_recon, nfft, nhop)
    #    spec_ang = torchaudio.functional.angle(spec_recon).unsqueeze(-1)
    #    spec_recon = torch.cat([mag * spec_ang.cos(), mag * spec_ang.sin()], dim=-1)
    #    t_recon = torchaudio.functional.istft(spec_recon, nfft, nhop)
    return t_recon


def save_audio(t, path, sr=22050, normalize=True):
    if normalize:
        scale = 1 / torch.abs(t).max()
        t *= scale
    torchaudio.save(path, t.unsqueeze(0), sr)


def load_resample_normalize(file_path, target_sr=22050):
    si, ei = torchaudio.info(file_path)
    bits = ei.bits_per_sample
    rate = 1 << (bits - 1)
    orig_sr = si.rate
    data, _ = torchaudio.load(file_path, normalization=rate)
    data = data.mean(keepdim=True, dim=0)
    rdata = kaldi.resample_waveform(data,orig_freq=orig_sr, new_freq=target_sr)
    return rdata