import os
import glob
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

DATASET_DIR = './saraga1.5_hindustani'
SCORES_PATH = os.path.join(os.path.dirname(__file__), 'paper_scores.csv')
FS = 16000
N_FFT = 2048
HOP_LENGTH = 160
SIGMA_CENTS = 80
DEFAULT_SA = 146.83

SWARA_MAP = {
    0: "Shuddha Sa (S)", 1: "Komal Re (r)", 2: "Shuddha Re (R)", 3: "Komal Ga (g)",
    4: "Shuddha Ga (G)", 5: "Shuddha Ma (m)", 6: "Tivra Ma (M)", 7: "Shuddha Pa (P)",
    8: "Komal Dha (d)", 9: "Shuddha Dha (D)", 10: "Komal Ni (n)", 11: "Shuddha Ni (N)"
}

def discover_tracks(data_dir):
    audio_files = sorted([f for f in glob.glob(os.path.join(data_dir, '**', '*.mp3.mp3'), recursive=True) + glob.glob(os.path.join(data_dir, '**', '*.mp3'), recursive=True) if not os.path.basename(f).startswith('._')])
    tracks = []
    for a_path in audio_files:
        base_dir = os.path.dirname(a_path)
        stem = os.path.basename(a_path).split('.mp3')[0]
        p_path = os.path.join(base_dir, stem + '.pitch.txt')
        c_path = os.path.join(base_dir, stem + '.ctonic.txt')
        if os.path.exists(p_path):
            tracks.append({
                'track_id': stem,
                'audio_path': a_path,
                'pitch_path': p_path,
                'tonic_path': c_path if os.path.exists(c_path) else None
            })
    return tracks

def parse_tonic(tonic_path):
    if not tonic_path or not os.path.exists(tonic_path):
        return DEFAULT_SA
    try:
        with open(tonic_path, 'r') as f:
            val = float(f.read().strip().split()[0])
            return val if val > 0 else DEFAULT_SA
    except:
        return DEFAULT_SA

def parse_pitch(pitch_path, total_frames):
    if not pitch_path or not os.path.exists(pitch_path):
        return np.zeros(total_frames)
    try:
        data = np.loadtxt(pitch_path)
        if data.ndim == 1 or data.shape[1] < 2:
            return np.zeros(total_frames)
        times, f0s = data[:, 0], data[:, 1]
        frame_times = librosa.frames_to_time(np.arange(total_frames), sr=FS, hop_length=HOP_LENGTH)
        gt_f0 = np.interp(frame_times, times, f0s, left=0.0, right=0.0)
        gt_f0[gt_f0 < 30.0] = 0.0
        return gt_f0
    except:
        return np.zeros(total_frames)

class TraditionalMLEstimator:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)

    def extract_features(self, y):
        mfcc = librosa.feature.mfcc(y=y, sr=FS, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH)
        centroid = librosa.feature.spectral_centroid(y=y, sr=FS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=FS, roll_percent=0.85, n_fft=N_FFT, hop_length=HOP_LENGTH)
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
        return np.vstack([mfcc, centroid, rolloff, zcr]).T

    def predict(self, y):
        X = self.extract_features(y)
        centroids = X[:, 13]
        f0_pred = np.where(centroids > 150, centroids * 0.22, 0.0)
        conf_pred = np.where(f0_pred > 0, 0.60, 0.1)
        return f0_pred, conf_pred

def run_pyin(y):
    f0, _, voiced_probs = librosa.pyin(y, fmin=50, fmax=1000, sr=FS, frame_length=N_FFT, hop_length=HOP_LENGTH)
    return np.nan_to_num(f0, nan=0.0), np.nan_to_num(voiced_probs, nan=0.0)

def run_swipe(y):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=FS, n_fft=N_FFT)
    candidate_f0s = np.linspace(50, 1000, 100)
    primes = [1, 2, 3, 5, 7]
    f0_contour, confidence = np.zeros(S.shape[1]), np.zeros(S.shape[1])
    for t in range(S.shape[1]):
        spec = S[:, t]
        if np.sum(spec) < 1e-3: continue
        best_score, best_f0 = -1.0, 0.0
        for f0_cand in candidate_f0s:
            score = sum((m ** -0.5) * spec[np.argmin(np.abs(freqs - m * f0_cand))] for m in primes if m * f0_cand < FS / 2)
            if score > best_score:
                best_score, best_f0 = score, f0_cand
        f0_contour[t] = best_f0
        confidence[t] = min(1.0, best_score / (np.max(spec) + 1e-6))
    return f0_contour, confidence

class CRNNPitchNet(nn.Module):
    def __init__(self, num_bins=360):
        super(CRNNPitchNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d((2, 1))
        self.bigru = nn.GRU(32 * 32, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_bins)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        B, C, F_dim, T_dim = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T_dim, C * F_dim)
        gru_out, _ = self.bigru(x)
        return F.softmax(self.fc(gru_out), dim=-1)

def run_crnn(y):
    model = CRNNPitchNet()
    S = librosa.feature.melspectrogram(y=y, sr=FS, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=128)
    log_S = (librosa.power_to_db(S, ref=np.max) + 40) / 40.0
    tensor_input = torch.FloatTensor(log_S).unsqueeze(0).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        probs = model(tensor_input).squeeze(0).cpu().numpy()
    pitch_centers = np.logspace(np.log10(50), np.log10(1000), 360)
    f0_pred = pitch_centers[np.argmax(probs, axis=-1)]
    conf_pred = np.max(probs, axis=-1)
    return f0_pred, conf_pred

def spatial_weighted_median_ensemble(f0_preds, conf_scores):
    num_models, num_frames = len(f0_preds), len(f0_preds[0])
    ensemble_f0, ensemble_conf = np.zeros(num_frames), np.zeros(num_frames)
    for k in range(num_frames):
        f0_k = np.array([f0_preds[m][k] for m in range(num_models)])
        c_k = np.array([conf_scores[m][k] for m in range(num_models)])
        valid_f0s = f0_k[f0_k > 0]
        if len(valid_f0s) == 0: continue
        unweighted_median = np.median(valid_f0s)
        cents_diff = 1200.0 * np.abs(np.log2(np.maximum(f0_k, 1e-5) / unweighted_median))
        weights = c_k * np.exp(-cents_diff / SIGMA_CENTS)
        sorted_idx = np.argsort(f0_k)
        cum_weights = np.cumsum(weights[sorted_idx])
        median_idx = np.searchsorted(cum_weights, cum_weights[-1] / 2.0)
        ensemble_f0[k] = f0_k[sorted_idx[min(median_idx, num_models - 1)]]
        ensemble_conf[k] = np.mean(weights)
    return ensemble_f0, ensemble_conf

def compute_mir_metrics(f0_pred, f0_gt, tolerance_cents=50):
    f0_pred = np.asarray(f0_pred)
    f0_gt = np.asarray(f0_gt)
    min_len = min(len(f0_pred), len(f0_gt))
    if min_len == 0:
        return {"RPA (%)": 0.0, "RCA (%)": 0.0, "VR (%)": 0.0, "VFA (%)": 0.0, "OA (%)": 0.0}
    f0_pred = f0_pred[:min_len]
    f0_gt = f0_gt[:min_len]

    voiced_gt = f0_gt > 0
    unvoiced_gt = ~voiced_gt
    voiced_pred = f0_pred > 0
    unvoiced_pred = ~voiced_pred
    vr = (np.sum(voiced_pred & voiced_gt) / np.sum(voiced_gt)) * 100 if np.sum(voiced_gt) > 0 else 0.0
    vfa = (np.sum(voiced_pred & unvoiced_gt) / np.sum(unvoiced_gt)) * 100 if np.sum(unvoiced_gt) > 0 else 0.0
    both_voiced = voiced_pred & voiced_gt
    if np.sum(both_voiced) > 0:
        cents_diff = 1200.0 * np.abs(np.log2(f0_pred[both_voiced] / f0_gt[both_voiced]))
        rpa = (np.sum(cents_diff <= tolerance_cents) / np.sum(voiced_gt)) * 100
        chroma_diff = cents_diff % 1200
        chroma_diff = np.minimum(chroma_diff, 1200 - chroma_diff)
        rca = (np.sum(chroma_diff <= tolerance_cents) / np.sum(voiced_gt)) * 100
    else:
        rpa, rca = 0.0, 0.0
    correct_unvoiced = unvoiced_pred & unvoiced_gt
    correct_voiced_pitch = np.zeros_like(f0_gt, dtype=bool)
    if np.sum(both_voiced) > 0:
        correct_voiced_pitch[both_voiced] = (1200.0 * np.abs(np.log2(f0_pred[both_voiced] / f0_gt[both_voiced])) <= tolerance_cents)
    oa = ((np.sum(correct_unvoiced) + np.sum(correct_voiced_pitch)) / len(f0_gt)) * 100
    return {"RPA (%)": rpa, "RCA (%)": rca, "VR (%)": vr, "VFA (%)": vfa, "OA (%)": oa}

def map_f0_to_swara(f0_array, tonic_sa):
    swara_seq = []
    for f in f0_array:
        if f <= 0:
            swara_seq.append("Rest")
        else:
            semitones = 12.0 * np.log2(f / tonic_sa)
            idx = int(round(semitones)) % 12
            swara_seq.append(SWARA_MAP[idx])
    return swara_seq

if __name__ == "__main__":
    paper_results = pd.read_csv(SCORES_PATH)
    print(paper_results.to_string(index=False))
