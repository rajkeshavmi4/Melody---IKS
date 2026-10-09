from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import librosa
import mirdata
import numpy as np
from scipy.ndimage import median_filter

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import librosa.display


TARGET_SR = 22050
HOP_LENGTH = 512
FMIN = librosa.note_to_hz("C2")
FMAX = librosa.note_to_hz("C7")
PITCH_TOLERANCE_CENTS = 50.0
PAPER_METHOD_VERSION = "paper_methods_2026_07_09_v1"
DEFAULT_GRAPH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saraga_hindustani_validation_graph.png",
)
DEFAULT_RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saraga_hindustani_validation_results.csv",
)
DEFAULT_ORCHSET_GRAPH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "orchset_validation_graph.png",
)
DEFAULT_ORCHSET_RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "orchset_validation_results.csv",
)
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".cache",
    "saraga_f0",
)
DEFAULT_SPECTROGRAM_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saraga_spectrograms",
)


class MethodUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SaragaTrack:
    track_id: str
    label: str
    rel_stem: str
    audio_path: Path
    pitch_path: Path
    tonic_path: Path | None
    metadata_path: Path | None

    @property
    def cache_id(self) -> str:
        digest = hashlib.sha1(self.rel_stem.encode("utf-8")).hexdigest()[:12]
        return f"{self.track_id[:12]}_{digest}".replace(os.sep, "_")


@dataclass
class F0Prediction:
    method: str
    times: np.ndarray
    frequencies: np.ndarray
    confidence: np.ndarray


def parse_partial_download(value):
    if value is None or value.strip() == "":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_methods(value: str) -> list[str]:
    methods = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not methods or methods == ["auto"]:
        return ["ensemble", "pyin", "salience", "swipe", "yin"]
    return methods


def hz_to_cents(frequencies: np.ndarray, reference_hz: float = 440.0) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=float)
    cents = np.full(frequencies.shape, np.nan, dtype=float)
    mask = frequencies > 0
    cents[mask] = 1200.0 * np.log2(frequencies[mask] / reference_hz)
    return cents


def cents_error(reference_frequencies, estimated_frequencies) -> np.ndarray:
    reference_cents = hz_to_cents(reference_frequencies)
    estimated_cents = hz_to_cents(estimated_frequencies)
    return np.abs(reference_cents - estimated_cents)


def chroma_cents_error(reference_frequencies, estimated_frequencies) -> np.ndarray:
    reference_chroma = np.mod(hz_to_cents(reference_frequencies), 1200.0)
    estimated_chroma = np.mod(hz_to_cents(estimated_frequencies), 1200.0)
    distance = np.abs(reference_chroma - estimated_chroma)
    return np.minimum(distance, 1200.0 - distance)


def normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.zeros_like(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return output
    lo = float(np.nanpercentile(values[finite], 5))
    hi = float(np.nanpercentile(values[finite], 95))
    if hi <= lo:
        hi = float(np.nanmax(values[finite]))
        lo = float(np.nanmin(values[finite]))
    if hi <= lo:
        output[finite] = 1.0
        return output
    output[finite] = np.clip((values[finite] - lo) / (hi - lo), 0.0, 1.0)
    return output


def frame_times(n_frames: int, sample_rate: int, hop_length: int) -> np.ndarray:
    return librosa.frames_to_time(np.arange(n_frames), sr=sample_rate, hop_length=hop_length)


def load_audio(audio_path: Path, sample_rate: int, max_seconds: float | None) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True,
        duration=max_seconds,
    )
    if audio.size == 0:
        raise RuntimeError(f"Audio file is empty or unreadable: {audio_path}")
    return audio.astype(np.float32, copy=False), sr


def infer_saraga_base(dataset_root: Path) -> Path:
    dataset_root = dataset_root.expanduser().resolve()
    nested = dataset_root / "saraga1.5_hindustani"
    if (nested / "file_paths.csv").exists():
        return nested
    return dataset_root


def read_metadata_label(metadata_path: Path | None, fallback: str) -> str:
    if metadata_path is None or not metadata_path.exists():
        return fallback
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        title = metadata.get("title") or fallback
        artists = metadata.get("artists") or []
        lead_artist = None
        for entry in artists:
            if entry.get("lead"):
                lead_artist = (entry.get("artist") or {}).get("name")
                break
        if lead_artist:
            return f"{title} - {lead_artist}"
        return title
    except Exception:
        return fallback


def load_local_saraga_tracks(dataset_root: str | Path) -> list[SaragaTrack]:
    base = infer_saraga_base(Path(dataset_root))
    file_paths = base / "file_paths.csv"
    tracks: list[SaragaTrack] = []

    if file_paths.exists():
        with file_paths.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rel_stem = row["filepath"].replace("/", os.sep)
                stem = base / rel_stem
                audio_path = Path(f"{stem}.mp3.mp3")
                pitch_path = Path(f"{stem}.pitch.txt")
                tonic_path = Path(f"{stem}.ctonic.txt")
                metadata_path = Path(f"{stem}.json")
                track_id = (row.get("mbid") or rel_stem).strip()
                label = read_metadata_label(metadata_path, Path(rel_stem).name)
                tracks.append(
                    SaragaTrack(
                        track_id=track_id,
                        label=label,
                        rel_stem=rel_stem,
                        audio_path=audio_path,
                        pitch_path=pitch_path,
                        tonic_path=tonic_path if tonic_path.exists() else None,
                        metadata_path=metadata_path if metadata_path.exists() else None,
                    )
                )
    else:
        for pitch_path in sorted(base.rglob("*.pitch.txt")):
            stem_text = str(pitch_path)[: -len(".pitch.txt")]
            stem = Path(stem_text)
            rel_stem = str(stem.relative_to(base))
            audio_path = Path(f"{stem}.mp3.mp3")
            tonic_path = Path(f"{stem}.ctonic.txt")
            metadata_path = Path(f"{stem}.json")
            label = read_metadata_label(metadata_path, stem.name)
            tracks.append(
                SaragaTrack(
                    track_id=hashlib.sha1(rel_stem.encode("utf-8")).hexdigest(),
                    label=label,
                    rel_stem=rel_stem,
                    audio_path=audio_path,
                    pitch_path=pitch_path,
                    tonic_path=tonic_path if tonic_path.exists() else None,
                    metadata_path=metadata_path if metadata_path.exists() else None,
                )
            )

    return tracks


def load_orchset_tracks(dataset_root: str | Path) -> list[SaragaTrack]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    audio_dir = dataset_root / "audio" / "mono"
    if not audio_dir.exists():
        audio_dir = dataset_root / "audio"
    gt_dir = dataset_root / "GT"
    tracks: list[SaragaTrack] = []

    if not audio_dir.exists() or not gt_dir.exists():
        return tracks

    for audio_path in sorted(audio_dir.glob("*.wav")):
        pitch_path = gt_dir / f"{audio_path.stem}.mel"
        if not pitch_path.exists():
            continue
        rel_stem = audio_path.relative_to(dataset_root).as_posix()
        tracks.append(
            SaragaTrack(
                track_id=audio_path.stem,
                label=audio_path.stem,
                rel_stem=rel_stem,
                audio_path=audio_path,
                pitch_path=pitch_path,
                tonic_path=None,
                metadata_path=None,
            )
        )

    return tracks


def load_pitch_annotation(track: SaragaTrack, max_seconds: float | None) -> tuple[np.ndarray, np.ndarray]:
    if not track.pitch_path.exists():
        raise FileNotFoundError(f"Missing pitch annotation: {track.pitch_path}")
    data = np.loadtxt(track.pitch_path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    times = data[:, 0]
    frequencies = data[:, 1]
    if max_seconds is not None:
        keep = times <= max_seconds
        times = times[keep]
        frequencies = frequencies[keep]
    return times.astype(float), frequencies.astype(float)


def load_tonic_hz(track: SaragaTrack) -> float | None:
    if track.tonic_path is None or not track.tonic_path.exists():
        return None
    try:
        value = float(track.tonic_path.read_text(encoding="utf-8").strip().split()[0])
    except Exception:
        return None
    if value > 0 and np.isfinite(value):
        return value
    return None


def align_prediction_to_times(
    reference_times: np.ndarray,
    predicted_times: np.ndarray,
    predicted_values: np.ndarray,
    max_time_distance: float,
) -> np.ndarray:
    reference_times = np.asarray(reference_times, dtype=float)
    predicted_times = np.asarray(predicted_times, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)
    aligned = np.zeros(reference_times.shape, dtype=float)
    if predicted_times.size == 0 or predicted_values.size == 0:
        return aligned

    right = np.searchsorted(predicted_times, reference_times, side="left")
    left = np.maximum(right - 1, 0)
    right = np.minimum(right, predicted_times.size - 1)
    left_distance = np.abs(reference_times - predicted_times[left])
    right_distance = np.abs(reference_times - predicted_times[right])
    use_right = right_distance < left_distance
    nearest = np.where(use_right, right, left)
    distance = np.minimum(left_distance, right_distance)
    valid = distance <= max_time_distance
    aligned[valid] = predicted_values[nearest[valid]]
    return aligned


def evaluate_f0_prediction(
    track: SaragaTrack,
    method: str,
    ground_truth_times: np.ndarray,
    ground_truth_frequencies: np.ndarray,
    prediction: F0Prediction,
    tolerance_cents: float,
) -> dict[str, float | str | int]:
    max_time_distance = float(np.median(np.diff(prediction.times)) * 0.75) if prediction.times.size > 1 else 0.05
    predicted_on_gt = align_prediction_to_times(
        ground_truth_times,
        prediction.times,
        prediction.frequencies,
        max_time_distance=max_time_distance,
    )

    voiced_reference = ground_truth_frequencies > 0
    voiced_estimate = predicted_on_gt > 0
    voiced_pairs = voiced_reference & voiced_estimate

    if not np.any(voiced_reference):
        raise RuntimeError("no voiced reference frames")

    pitch_correct = np.zeros_like(voiced_reference, dtype=bool)
    chroma_correct = np.zeros_like(voiced_reference, dtype=bool)
    if np.any(voiced_pairs):
        pitch_correct[voiced_pairs] = (
            cents_error(ground_truth_frequencies[voiced_pairs], predicted_on_gt[voiced_pairs])
            <= tolerance_cents
        )
        chroma_correct[voiced_pairs] = (
            chroma_cents_error(ground_truth_frequencies[voiced_pairs], predicted_on_gt[voiced_pairs])
            <= tolerance_cents
        )

    unvoiced_reference = ~voiced_reference
    raw_pitch_accuracy = float(np.sum(pitch_correct) / np.sum(voiced_reference))
    raw_chroma_accuracy = float(np.sum(chroma_correct) / np.sum(voiced_reference))
    voicing_recall = float(np.sum(voiced_pairs) / np.sum(voiced_reference))
    if np.any(unvoiced_reference):
        voicing_false_alarm = float(np.sum(voiced_estimate & unvoiced_reference) / np.sum(unvoiced_reference))
    else:
        voicing_false_alarm = 0.0
    overall_accuracy = float(np.sum(pitch_correct | (~voiced_reference & ~voiced_estimate)) / len(voiced_reference))

    return {
        "track_id": track.track_id,
        "track_label": track.label,
        "method": method,
        "voiced_reference_frames": int(np.sum(voiced_reference)),
        "raw_pitch_accuracy": raw_pitch_accuracy,
        "raw_chroma_accuracy": raw_chroma_accuracy,
        "voicing_recall": voicing_recall,
        "voicing_false_alarm": voicing_false_alarm,
        "overall_accuracy": overall_accuracy,
    }


def contour_cleanup(
    frequencies: np.ndarray,
    confidence: np.ndarray,
    min_voiced_frames: int = 4,
    max_gap_frames: int = 2,
) -> np.ndarray:
    cleaned = np.asarray(frequencies, dtype=float).copy()
    voiced = cleaned > 0
    if voiced.size == 0:
        return cleaned

    changes = np.flatnonzero(np.diff(np.r_[False, voiced, False]))
    starts = changes[0::2]
    ends = changes[1::2]
    for start, end in zip(starts, ends):
        if end - start < min_voiced_frames:
            cleaned[start:end] = 0.0

    voiced = cleaned > 0
    changes = np.flatnonzero(np.diff(np.r_[True, voiced, True]))
    starts = changes[0::2]
    ends = changes[1::2]
    for start, end in zip(starts, ends):
        if 0 < start and end < len(cleaned) and end - start <= max_gap_frames:
            if confidence[start:end].mean() > 0.15:
                cleaned[start:end] = np.interp(
                    np.arange(start, end),
                    [start - 1, end],
                    [cleaned[start - 1], cleaned[end]],
                )
    return cleaned


def octave_stabilize(frequencies: np.ndarray, tonic_hz: float | None, fmin: float, fmax: float) -> np.ndarray:
    stabilized = np.asarray(frequencies, dtype=float).copy()
    previous = 0.0
    lower = fmin
    upper = fmax
    if tonic_hz is not None:
        lower = max(fmin, tonic_hz * 2.0 ** -1.25)
        upper = min(fmax, tonic_hz * 2.0 ** 3.25)

    for index, value in enumerate(stabilized):
        if value <= 0:
            previous = 0.0
            continue
        candidates = value * (2.0 ** np.arange(-2, 3))
        candidates = candidates[(candidates >= lower) & (candidates <= upper)]
        if candidates.size == 0:
            stabilized[index] = 0.0
            previous = 0.0
            continue

        if lower <= value <= upper:
            selected = float(value)
            if previous > 0:
                raw_jump = float(cents_error(previous, value))
                jump_costs = cents_error(previous, candidates)
                octave_shift_penalty = 1300.0 * np.abs(np.log2(candidates / value))
                best_index = int(np.argmin(jump_costs + octave_shift_penalty))
                best_value = float(candidates[best_index])
                best_cost = float(jump_costs[best_index] + octave_shift_penalty[best_index])
                if best_cost < raw_jump:
                    selected = best_value
            stabilized[index] = selected
        else:
            stabilized[index] = float(candidates[int(np.argmin(np.abs(hz_to_cents(candidates) - hz_to_cents(value))))])
        previous = stabilized[index]
    return stabilized


def postprocess_prediction(
    prediction: F0Prediction,
    tonic_hz: float | None,
    fmin: float,
    fmax: float,
    median_size: int = 3,
) -> F0Prediction:
    frequencies = np.nan_to_num(prediction.frequencies, nan=0.0, posinf=0.0, neginf=0.0)
    confidence = np.nan_to_num(prediction.confidence, nan=0.0, posinf=0.0, neginf=0.0)
    frequencies[(frequencies < fmin) | (frequencies > fmax)] = 0.0
    frequencies[confidence < 0.03] = 0.0
    frequencies = contour_cleanup(frequencies, confidence)
    frequencies = octave_stabilize(frequencies, tonic_hz, fmin, fmax)

    if median_size > 1 and len(frequencies) >= median_size:
        voiced = frequencies > 0
        smoothed = frequencies.copy()
        smoothed[voiced] = median_filter(frequencies, size=median_size, mode="nearest")[voiced]
        frequencies = smoothed

    return F0Prediction(
        method=prediction.method,
        times=np.asarray(prediction.times, dtype=float),
        frequencies=frequencies,
        confidence=confidence,
    )


def predict_pyin(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> F0Prediction:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sample_rate,
            hop_length=hop_length,
            frame_length=2048,
        )
    frequencies = np.nan_to_num(f0, nan=0.0)
    if voiced_flag is not None:
        frequencies[~voiced_flag] = 0.0
    confidence = np.nan_to_num(voiced_prob, nan=0.0)
    return F0Prediction("pyin", frame_times(len(frequencies), sample_rate, hop_length), frequencies, confidence)


def predict_yin(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> F0Prediction:
    frame_length = 2048
    frequencies = librosa.yin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        hop_length=hop_length,
        frame_length=frame_length,
        trough_threshold=0.10,
    )
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms = rms[: len(frequencies)]
    confidence = normalize_positive(rms)
    if rms.size:
        energy_floor = max(float(np.nanpercentile(rms, 15)), float(np.nanmax(rms)) * 0.025)
        frequencies[rms < energy_floor] = 0.0
    frequencies = np.nan_to_num(frequencies, nan=0.0)
    return F0Prediction("yin", frame_times(len(frequencies), sample_rate, hop_length), frequencies, confidence)


def prime_harmonics(max_harmonic: int) -> np.ndarray:
    if max_harmonic < 1:
        return np.array([], dtype=int)
    harmonics = [1]
    for candidate in range(2, max_harmonic + 1):
        is_prime = True
        for divisor in range(2, int(math.sqrt(candidate)) + 1):
            if candidate % divisor == 0:
                is_prime = False
                break
        if is_prime:
            harmonics.append(candidate)
    return np.asarray(sorted(set(harmonics)), dtype=int)


def log_frequency_grid(fmin: float, fmax: float, bins_per_octave: int) -> np.ndarray:
    n_bins = int(np.floor(bins_per_octave * np.log2(fmax / fmin))) + 1
    return fmin * (2.0 ** (np.arange(n_bins) / bins_per_octave))


def nearest_frequency_bins(frequencies_hz: np.ndarray, frequency_axis: np.ndarray) -> np.ndarray:
    indexes = np.searchsorted(frequency_axis, frequencies_hz, side="left")
    indexes = np.clip(indexes, 1, len(frequency_axis) - 1)
    left = indexes - 1
    choose_left = np.abs(frequency_axis[left] - frequencies_hz) <= np.abs(frequency_axis[indexes] - frequencies_hz)
    return np.where(choose_left, left, indexes)


def stft_magnitude(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window="hann"))
    frequency_axis = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    return spectrum.astype(np.float32, copy=False), frequency_axis


def dynamic_voicing_threshold(confidence: np.ndarray, floor: float) -> float:
    positive = confidence[np.isfinite(confidence) & (confidence > 0)]
    if positive.size == 0:
        return floor
    return max(floor, float(np.nanpercentile(positive, 30)) * 0.75)


def select_from_score_grid(
    scores: np.ndarray,
    candidates: np.ndarray,
    voicing_floor: float,
    use_viterbi: bool,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    scores[~np.isfinite(scores)] = -np.inf
    if scores.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if use_viterbi and scores.shape[1] <= 20000:
        best_bins = viterbi_decode_scores(scores, max_jump_bins=18, transition_penalty=0.035)
    else:
        best_bins = np.argmax(scores, axis=0)
        if best_bins.size >= 5:
            best_bins = median_filter(best_bins, size=5, mode="nearest")

    frame_indexes = np.arange(scores.shape[1])
    confidence = scores[best_bins, frame_indexes]
    confidence = normalize_positive(confidence)
    threshold = dynamic_voicing_threshold(confidence, voicing_floor)
    frequencies = candidates[best_bins].astype(float)

    # Local weighted average, matching the dYIN/dSWIPE paper's soft decoding idea.
    for frame_index, bin_index in enumerate(best_bins):
        lo = max(0, int(bin_index) - 1)
        hi = min(len(candidates), int(bin_index) + 2)
        local_scores = scores[lo:hi, frame_index]
        if np.all(~np.isfinite(local_scores)):
            continue
        local_scores = local_scores - np.nanmax(local_scores)
        weights = np.exp(local_scores)
        if np.sum(weights) > 0:
            frequencies[frame_index] = float(np.sum(candidates[lo:hi] * weights) / np.sum(weights))

    frequencies[confidence < threshold] = 0.0
    return frequencies, confidence


def viterbi_decode_scores(scores: np.ndarray, max_jump_bins: int, transition_penalty: float) -> np.ndarray:
    n_bins, n_frames = scores.shape
    log_scores = scores - np.nanmax(scores, axis=0, keepdims=True)
    log_scores = np.nan_to_num(log_scores, nan=-50.0, neginf=-50.0, posinf=0.0)
    backpointers = np.zeros((n_bins, n_frames), dtype=np.int16)
    previous = log_scores[:, 0].astype(np.float32)

    for frame_index in range(1, n_frames):
        current = np.empty(n_bins, dtype=np.float32)
        for bin_index in range(n_bins):
            lo = max(0, bin_index - max_jump_bins)
            hi = min(n_bins, bin_index + max_jump_bins + 1)
            candidates = previous[lo:hi] - transition_penalty * np.abs(np.arange(lo, hi) - bin_index)
            offset = int(np.argmax(candidates))
            backpointers[bin_index, frame_index] = lo + offset
            current[bin_index] = candidates[offset] + log_scores[bin_index, frame_index]
        previous = current

    path = np.zeros(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(previous))
    for frame_index in range(n_frames - 1, 0, -1):
        path[frame_index - 1] = int(backpointers[path[frame_index], frame_index])
    return path


def predict_swipe(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> F0Prediction:
    n_fft = 4096
    magnitude, frequency_axis = stft_magnitude(audio, sample_rate, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.log1p(magnitude)
    candidates = log_frequency_grid(fmin, fmax, bins_per_octave=60)
    nyquist = sample_rate / 2.0
    frame_norm = np.sqrt(np.sum(magnitude * magnitude, axis=0) + 1e-8)
    scores = np.empty((len(candidates), magnitude.shape[1]), dtype=np.float32)

    for candidate_index, f0 in enumerate(candidates):
        max_harmonic = int(min(18, math.floor(nyquist / f0)))
        harmonics = prime_harmonics(max_harmonic)
        if harmonics.size == 0:
            scores[candidate_index] = -np.inf
            continue
        harmonic_freqs = harmonics * f0
        peak_bins = nearest_frequency_bins(harmonic_freqs, frequency_axis)
        left_bins = nearest_frequency_bins(np.maximum((harmonics - 0.5) * f0, frequency_axis[1]), frequency_axis)
        right_bins = nearest_frequency_bins(np.minimum((harmonics + 0.5) * f0, nyquist), frequency_axis)
        weights = 1.0 / np.sqrt(harmonics.astype(float))
        peak_energy = magnitude[peak_bins]
        valley_energy = 0.5 * (magnitude[left_bins] + magnitude[right_bins])
        score = np.sum(weights[:, None] * (peak_energy - valley_energy), axis=0)
        scores[candidate_index] = score / (np.sqrt(np.sum(weights * weights)) * frame_norm)

    frequencies, confidence = select_from_score_grid(
        scores,
        candidates,
        voicing_floor=0.08,
        use_viterbi=True,
    )
    return F0Prediction("swipe", frame_times(len(frequencies), sample_rate, hop_length), frequencies, confidence)


def predict_salience(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> F0Prediction:
    harmonic_audio = librosa.effects.harmonic(audio, margin=4.0)
    n_fft = 4096
    magnitude, frequency_axis = stft_magnitude(harmonic_audio, sample_rate, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.log1p(magnitude)
    magnitude -= np.nanpercentile(magnitude, 20, axis=0, keepdims=True)
    magnitude = np.maximum(magnitude, 0.0)

    candidates = log_frequency_grid(fmin, fmax, bins_per_octave=60)
    nyquist = sample_rate / 2.0
    scores = np.zeros((len(candidates), magnitude.shape[1]), dtype=np.float32)
    harmonic_weights = np.array([1.0, 0.75, 0.58, 0.45, 0.35, 0.28, 0.22, 0.18], dtype=np.float32)

    for candidate_index, f0 in enumerate(candidates):
        harmonic_numbers = np.arange(1, len(harmonic_weights) + 1)
        harmonic_freqs = harmonic_numbers * f0
        valid = harmonic_freqs <= nyquist
        if not np.any(valid):
            scores[candidate_index] = -np.inf
            continue
        bins = nearest_frequency_bins(harmonic_freqs[valid], frequency_axis)
        weights = harmonic_weights[valid]
        scores[candidate_index] = np.sum(weights[:, None] * magnitude[bins], axis=0) / np.sum(weights)

    frequencies, confidence = select_from_score_grid(
        scores,
        candidates,
        voicing_floor=0.12,
        use_viterbi=True,
    )
    return F0Prediction("salience", frame_times(len(frequencies), sample_rate, hop_length), frequencies, confidence)


def predict_crepe(
    audio_path: Path,
    max_seconds: float | None,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> F0Prediction:
    try:
        import crepe
    except Exception as exc:
        raise MethodUnavailable("CREPE is not installed for this Python environment.") from exc

    audio, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True, duration=max_seconds)
    step_size_ms = max(5, int(round(1000.0 * hop_length / TARGET_SR)))
    times, frequencies, confidence, _activation = crepe.predict(
        audio,
        sample_rate,
        step_size=step_size_ms,
        viterbi=True,
        verbose=0,
    )
    frequencies = np.asarray(frequencies, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    frequencies[(frequencies < fmin) | (frequencies > fmax)] = 0.0
    return F0Prediction("crepe", np.asarray(times, dtype=float), frequencies, confidence)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = 0.5 * cumulative[-1]
    return float(values[int(np.searchsorted(cumulative, cutoff))])


def predict_ensemble(
    track: SaragaTrack,
    args,
    tonic_hz: float | None,
    base_methods: tuple[str, ...] = ("pyin", "salience", "swipe", "yin"),
) -> F0Prediction:
    base_predictions = []
    unavailable = []
    for method in base_methods:
        try:
            base_predictions.append(get_prediction(track, method, args, tonic_hz))
        except MethodUnavailable as exc:
            unavailable.append(str(exc))

    if not base_predictions:
        raise MethodUnavailable("; ".join(unavailable) or "No ensemble backends are available.")

    reference = base_predictions[0]
    times = reference.times
    all_frequencies = []
    all_confidences = []
    max_distance = float(np.median(np.diff(times)) * 0.75) if len(times) > 1 else 0.05

    for prediction in base_predictions:
        all_frequencies.append(
            align_prediction_to_times(times, prediction.times, prediction.frequencies, max_time_distance=max_distance)
        )
        all_confidences.append(
            align_prediction_to_times(times, prediction.times, prediction.confidence, max_time_distance=max_distance)
        )

    frequency_matrix = np.vstack(all_frequencies)
    confidence_matrix = np.vstack(all_confidences)
    frequencies = np.zeros(len(times), dtype=float)
    confidence = np.zeros(len(times), dtype=float)

    for frame_index in range(len(times)):
        frame_freqs = frequency_matrix[:, frame_index]
        frame_conf = confidence_matrix[:, frame_index]
        voiced = (frame_freqs > 0) & (frame_conf > 0.05)
        if not np.any(voiced):
            continue
        voiced_freqs = frame_freqs[voiced]
        voiced_conf = frame_conf[voiced]
        if voiced_freqs.size == 1:
            frequencies[frame_index] = float(voiced_freqs[0])
            confidence[frame_index] = float(voiced_conf[0])
            continue
        cents = hz_to_cents(voiced_freqs)
        median_cents = weighted_median(cents, voiced_conf)
        agreement = np.exp(-np.minimum(np.abs(cents - median_cents), 1200.0) / 80.0)
        weights = voiced_conf * agreement
        if np.sum(weights) <= 0:
            weights = voiced_conf
        fused_cents = weighted_median(cents, weights)
        frequencies[frame_index] = 440.0 * (2.0 ** (fused_cents / 1200.0))
        confidence[frame_index] = float(np.clip(np.mean(voiced_conf) * (0.5 + 0.5 * np.mean(agreement)), 0.0, 1.0))

    return F0Prediction("ensemble", times, frequencies, confidence)


def prediction_signature(track: SaragaTrack, method: str, args) -> str:
    audio_stat = track.audio_path.stat() if track.audio_path.exists() else None
    payload = {
        "version": PAPER_METHOD_VERSION,
        "method": method,
        "track": track.rel_stem,
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "max_seconds": args.max_seconds,
        "audio_size": audio_stat.st_size if audio_stat else None,
        "audio_mtime": int(audio_stat.st_mtime) if audio_stat else None,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def prediction_cache_path(track: SaragaTrack, method: str, args) -> Path:
    signature = prediction_signature(track, method, args)
    return Path(args.cache_dir) / method / f"{track.cache_id}_{signature}.npz"


def read_cached_prediction(path: Path, method: str) -> F0Prediction | None:
    if not path.exists():
        return None
    data = np.load(path)
    return F0Prediction(
        method=method,
        times=data["times"],
        frequencies=data["frequencies"],
        confidence=data["confidence"],
    )


def write_cached_prediction(path: Path, prediction: F0Prediction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times=prediction.times,
        frequencies=prediction.frequencies,
        confidence=prediction.confidence,
    )


def compute_prediction(track: SaragaTrack, method: str, args, tonic_hz: float | None) -> F0Prediction:
    if method == "ensemble":
        prediction = predict_ensemble(track, args, tonic_hz)
        return postprocess_prediction(prediction, tonic_hz, args.fmin, args.fmax)
    if method == "crepe":
        prediction = predict_crepe(track.audio_path, args.max_seconds, args.hop_length, args.fmin, args.fmax)
        return postprocess_prediction(prediction, tonic_hz, args.fmin, args.fmax)

    audio, sample_rate = load_audio(track.audio_path, args.sample_rate, args.max_seconds)
    if method == "pyin":
        prediction = predict_pyin(audio, sample_rate, args.hop_length, args.fmin, args.fmax)
    elif method in {"yin", "dyin"}:
        prediction = predict_yin(audio, sample_rate, args.hop_length, args.fmin, args.fmax)
        prediction.method = method
    elif method in {"swipe", "dswipe"}:
        prediction = predict_swipe(audio, sample_rate, args.hop_length, args.fmin, args.fmax)
        prediction.method = method
    elif method in {"salience", "deepsalience", "multipitch"}:
        prediction = predict_salience(audio, sample_rate, args.hop_length, args.fmin, args.fmax)
        prediction.method = method
    else:
        raise MethodUnavailable(f"Unknown method: {method}")

    return postprocess_prediction(prediction, tonic_hz, args.fmin, args.fmax)


def get_prediction(track: SaragaTrack, method: str, args, tonic_hz: float | None) -> F0Prediction:
    cache_path = prediction_cache_path(track, method, args)
    if args.use_cache and not args.force_recompute:
        cached = read_cached_prediction(cache_path, method)
        if cached is not None:
            return cached

    prediction = compute_prediction(track, method, args, tonic_hz)
    if args.use_cache:
        write_cached_prediction(cache_path, prediction)
    return prediction


def summarize_results(results: list[dict[str, float | str | int]]) -> list[dict[str, float | str | int]]:
    methods = sorted({str(result["method"]) for result in results})
    summary = []
    for method in methods:
        method_results = [result for result in results if result["method"] == method]
        summary.append(
            {
                "method": method,
                "tracks": len(method_results),
                "mean_raw_pitch_accuracy": float(np.mean([r["raw_pitch_accuracy"] for r in method_results])),
                "mean_raw_chroma_accuracy": float(np.mean([r["raw_chroma_accuracy"] for r in method_results])),
                "mean_voicing_recall": float(np.mean([r["voicing_recall"] for r in method_results])),
                "mean_voicing_false_alarm": float(np.mean([r["voicing_false_alarm"] for r in method_results])),
                "mean_overall_accuracy": float(np.mean([r["overall_accuracy"] for r in method_results])),
            }
        )
    summary.sort(key=lambda item: (item["mean_overall_accuracy"], item["mean_raw_pitch_accuracy"]), reverse=True)
    return summary


def write_results_csv(results: list[dict[str, float | str | int]], output_path: str | None) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track_id",
        "track_label",
        "method",
        "voiced_reference_frames",
        "raw_pitch_accuracy",
        "raw_chroma_accuracy",
        "voicing_recall",
        "voicing_false_alarm",
        "overall_accuracy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV results to {path}")


def safe_filename(value: str, max_length: int = 90) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            cleaned.append(char)
        elif char.isspace():
            cleaned.append("_")
    name = "".join(cleaned).strip("._")
    return (name or "track")[:max_length]


def generate_spectrogram_plot(
    track: SaragaTrack,
    prediction: F0Prediction,
    ground_truth_times: np.ndarray,
    ground_truth_frequencies: np.ndarray,
    args,
) -> Path:
    output_dir = Path(args.spectrogram_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio, sample_rate = load_audio(track.audio_path, args.sample_rate, args.max_seconds)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=2048,
        hop_length=args.hop_length,
        n_mels=160,
        fmin=max(20.0, args.fmin * 0.5),
        fmax=min(sample_rate / 2.0, args.fmax * 1.5),
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    figure, axis = plt.subplots(figsize=(16, 6))
    librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=args.hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=max(20.0, args.fmin * 0.5),
        fmax=min(sample_rate / 2.0, args.fmax * 1.5),
        cmap="magma",
        ax=axis,
    )

    gt_mask = ground_truth_frequencies > 0
    pred_mask = prediction.frequencies > 0
    axis.plot(
        ground_truth_times[gt_mask],
        ground_truth_frequencies[gt_mask],
        color="#22c55e",
        linewidth=1.5,
        label="Saraga annotation",
    )
    axis.plot(
        prediction.times[pred_mask],
        prediction.frequencies[pred_mask],
        color="#38bdf8",
        linewidth=1.2,
        alpha=0.9,
        label=f"{prediction.method} estimate",
    )
    axis.set_title(f"{track.label} - {prediction.method}")
    axis.set_ylim(max(20.0, args.fmin * 0.5), min(sample_rate / 2.0, args.fmax * 1.5))
    if args.max_seconds is not None:
        axis.set_xlim(0.0, args.max_seconds)
    axis.legend(loc="upper right")
    figure.tight_layout()

    output_path = output_dir / f"{track.cache_id}_{safe_filename(prediction.method)}_{safe_filename(track.label)}.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_validation_graph(results, output_path=DEFAULT_GRAPH_PATH, dataset_name: str = "saraga"):
    if not results:
        return

    summary = summarize_results(results)
    best_method = str(summary[0]["method"])
    best_results = [result for result in results if result["method"] == best_method]
    track_labels = [str(result["track_label"]) for result in best_results]
    rpa_values = [float(result["raw_pitch_accuracy"]) for result in best_results]
    rca_values = [float(result["raw_chroma_accuracy"]) for result in best_results]

    figure, axes = plt.subplots(2, 1, figsize=(18, 10))

    x_values = np.arange(len(best_results))
    axes[0].plot(x_values, rpa_values, marker="o", linewidth=1.4, color="#0f766e", label="RPA")
    axes[0].plot(x_values, rca_values, marker="o", linewidth=1.1, color="#c2410c", label="RCA")
    axes[0].axhline(float(np.mean(rpa_values)), color="#115e59", linestyle="--", linewidth=1.2)
    title_prefix = "Saraga Hindustani" if dataset_name.lower() == "saraga" else "ORCHSET"
    axes[0].set_title(f"Best {title_prefix} Method by Track: {best_method}")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="lower right")
    tick_step = max(1, len(track_labels) // 16)
    tick_positions = x_values[::tick_step]
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels([track_labels[index] for index in tick_positions], rotation=45, ha="right", fontsize=8)

    methods = [str(item["method"]) for item in summary]
    mean_rpa = [float(item["mean_raw_pitch_accuracy"]) for item in summary]
    mean_rca = [float(item["mean_raw_chroma_accuracy"]) for item in summary]
    mean_oa = [float(item["mean_overall_accuracy"]) for item in summary]
    positions = np.arange(len(methods))
    width = 0.25
    axes[1].bar(positions - width, mean_rpa, width=width, color="#0f766e", label="Mean RPA")
    axes[1].bar(positions, mean_rca, width=width, color="#c2410c", label="Mean RCA")
    axes[1].bar(positions + width, mean_oa, width=width, color="#1d4ed8", label="Mean OA")
    axes[1].set_title("Method Comparison")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(methods, rotation=25, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="lower right")

    figure.suptitle(f"{title_prefix} Paper-Inspired F0 Evaluation", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=[0, 0.03, 1, 0.96])
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved validation graph to {output_path}")


def download_saraga_hindustani_dataset(local_dataset_path, partial_download=None, force_overwrite=False):
    dataset = mirdata.initialize("saraga_hindustani", data_home=local_dataset_path)
    if partial_download is None:
        print("Downloading Saraga Hindustani dataset via mirdata...")
    else:
        print(f"Downloading Saraga Hindustani dataset via mirdata (partial: {partial_download})...")
    try:
        dataset.download(partial_download=partial_download, force_overwrite=force_overwrite)
    except OSError as error:
        message = str(error)
        if "MD5 checksum" not in message and "corrupted" not in message:
            raise

        zip_path = os.path.join(local_dataset_path, "saraga1.5_hindustani.zip")
        if os.path.exists(zip_path):
            print(f"Removing corrupted archive and retrying download: {zip_path}")
            os.remove(zip_path)

        dataset.download(partial_download=partial_download, force_overwrite=True)
    return mirdata.initialize("saraga_hindustani", data_home=local_dataset_path)


def filter_tracks(tracks: list[SaragaTrack], args) -> list[SaragaTrack]:
    if args.track_filter:
        needle = args.track_filter.lower()
        tracks = [
            track
            for track in tracks
            if needle in track.rel_stem.lower() or needle in track.label.lower() or needle in track.track_id.lower()
        ]
    if args.limit is not None:
        tracks = tracks[: args.limit]
    return tracks


def run_evaluation(args) -> tuple[list[dict[str, float | str | int]], list[dict[str, float | str | int]]]:
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    dataset_type = (args.dataset_type or "saraga").lower()
    dataset_root = Path(args.dataset_root) if args.dataset_root else Path(workspace_root) / (
        "orchset" if dataset_type == "orchset" else "saraga_hindustani"
    )
    if args.download:
        dataset_root.mkdir(parents=True, exist_ok=True)
        download_saraga_hindustani_dataset(
            str(dataset_root),
            partial_download=parse_partial_download(args.partial_download),
            force_overwrite=args.force_overwrite,
        )

    if dataset_type == "orchset":
        tracks = load_orchset_tracks(dataset_root)
        dataset_label = "ORCHSET"
        if not tracks:
            raise RuntimeError(f"No ORCHSET tracks found under {dataset_root}")
    else:
        tracks = load_local_saraga_tracks(dataset_root)
        dataset_label = "Saraga"
        if not tracks:
            raise RuntimeError(f"No Saraga tracks found under {dataset_root}")

    tracks = filter_tracks(tracks, args)
    if not tracks:
        raise RuntimeError(f"No {dataset_label} tracks found under {dataset_root}")

    methods = parse_methods(args.methods)
    print(f"Dataset root: {dataset_root}")
    print(f"Tracks selected: {len(tracks)}")
    print(f"Methods: {', '.join(methods)}")
    if args.max_seconds is not None:
        print(f"Evaluating only the first {args.max_seconds:g} seconds of each track.")

    results = []
    skipped = []
    for track_number, track in enumerate(tracks, start=1):
        print(f"\n[{track_number}/{len(tracks)}] {track.label}")
        if not track.audio_path.exists():
            skipped.append({"track_id": track.track_id, "method": "all", "reason": "missing audio"})
            print("  skipped: missing audio")
            continue
        if not track.pitch_path.exists():
            skipped.append({"track_id": track.track_id, "method": "all", "reason": "missing pitch annotation"})
            print("  skipped: missing pitch annotation")
            continue

        try:
            gt_times, gt_frequencies = load_pitch_annotation(track, args.max_seconds)
        except Exception as exc:
            skipped.append({"track_id": track.track_id, "method": "all", "reason": str(exc)})
            print(f"  skipped: {exc}")
            continue
        tonic_hz = load_tonic_hz(track)

        for method in methods:
            try:
                prediction = get_prediction(track, method, args, tonic_hz)
                result = evaluate_f0_prediction(
                    track,
                    method,
                    gt_times,
                    gt_frequencies,
                    prediction,
                    args.pitch_tolerance_cents,
                )
            except MethodUnavailable as exc:
                skipped.append({"track_id": track.track_id, "method": method, "reason": str(exc)})
                print(f"  {method}: unavailable ({exc})")
                continue
            except Exception as exc:
                skipped.append({"track_id": track.track_id, "method": method, "reason": str(exc)})
                print(f"  {method}: failed ({exc})")
                continue

            results.append(result)
            print(
                "  {method}: RPA={rpa:.4f} RCA={rca:.4f} VR={vr:.4f} VFA={vfa:.4f} OA={oa:.4f}".format(
                    method=method,
                    rpa=result["raw_pitch_accuracy"],
                    rca=result["raw_chroma_accuracy"],
                    vr=result["voicing_recall"],
                    vfa=result["voicing_false_alarm"],
                    oa=result["overall_accuracy"],
                )
            )
            if args.save_spectrograms:
                try:
                    spectrogram_path = generate_spectrogram_plot(track, prediction, gt_times, gt_frequencies, args)
                    print(f"    spectrogram: {spectrogram_path}")
                except Exception as exc:
                    print(f"    spectrogram failed: {exc}")

    if not results:
        print("No tracks were evaluated. Check that the audio and pitch annotation files are present.")
        return results, skipped

    print("\nSummary")
    summary = summarize_results(results)
    for row in summary:
        print(
            "  {method}: tracks={tracks}, mean RPA={rpa:.4f}, mean RCA={rca:.4f}, mean VR={vr:.4f}, "
            "mean VFA={vfa:.4f}, mean OA={oa:.4f}".format(
                method=row["method"],
                tracks=row["tracks"],
                rpa=row["mean_raw_pitch_accuracy"],
                rca=row["mean_raw_chroma_accuracy"],
                vr=row["mean_voicing_recall"],
                vfa=row["mean_voicing_false_alarm"],
                oa=row["mean_overall_accuracy"],
            )
        )
    best = summary[0]
    print(f"\nBest overall by mean OA: {best['method']} ({best['mean_overall_accuracy']:.4f})")
    print(f"Skipped method-track pairs: {len(skipped)}")

    csv_output = args.csv_output
    if not csv_output:
        csv_output = DEFAULT_ORCHSET_RESULTS_CSV if dataset_type == "orchset" else DEFAULT_RESULTS_CSV
    graph_output = args.graph_output
    if not graph_output:
        graph_output = DEFAULT_ORCHSET_GRAPH_PATH if dataset_type == "orchset" else DEFAULT_GRAPH_PATH

    write_results_csv(results, csv_output)
    if graph_output:
        generate_validation_graph(results, output_path=graph_output, dataset_name=dataset_type)
    return results, skipped


def run_mirdata_evaluation_loop(graph_output_path=DEFAULT_GRAPH_PATH, partial_download=None, force_overwrite=False):
    args = build_arg_parser().parse_args([])
    args.graph_output = graph_output_path
    args.partial_download = ",".join(partial_download or []) if partial_download else None
    args.force_overwrite = force_overwrite
    return run_evaluation(args)


def build_arg_parser() -> argparse.ArgumentParser:
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paper-inspired F0/melody extraction methods on Saraga Hindustani. "
            "Implemented methods include pYIN/YIN, SWIPE/SWIPE'-style harmonic templates, "
            "DeepSalience/Melodia-style harmonic salience with Viterbi smoothing, optional CREPE, "
            "and a tonic-aware ensemble."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Path containing the dataset files. Defaults to orchset/ for ORCHSET and saraga_hindustani/ for Saraga.",
    )
    parser.add_argument(
        "--dataset-type",
        default="saraga",
        choices=["saraga", "orchset"],
        help="Dataset family to evaluate: saraga or orchset.",
    )
    parser.add_argument(
        "--methods",
        default="ensemble,pyin,salience,swipe,yin",
        help=(
            "Comma-separated methods. Use auto for the default set. Available: ensemble, pyin, yin, dyin, "
            "swipe, dswipe, salience, deepsalience, multipitch, crepe."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N selected tracks.")
    parser.add_argument("--track-filter", default=None, help="Substring filter over track path, title, or id.")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Evaluate  only the first N seconds per track. Useful for quick method sweeps.",
    )
    parser.add_argument("--sample-rate", type=int, default=TARGET_SR, help="Analysis sample rate.")
    parser.add_argument("--hop-length", type=int, default=HOP_LENGTH, help="Analysis hop length in samples.")
    parser.add_argument("--fmin", type=float, default=FMIN, help="Minimum F0 in Hz.")
    parser.add_argument("--fmax", type=float, default=FMAX, help="Maximum F0 in Hz.")
    parser.add_argument(
        "--pitch-tolerance-cents",
        type=float,
        default=PITCH_TOLERANCE_CENTS,
        help="Tolerance for raw pitch/chroma accuracy.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Directory for cached method predictions.",
    )
    parser.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Disable prediction caching.",
    )
    parser.set_defaults(use_cache=True)
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore matching cached predictions and recompute them.",
    )
    parser.add_argument(
        "--graph-output",
        default=None,
        help="Path for the saved validation graph PNG. Use an empty string to skip.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Path for per-track CSV metrics. Use an empty string to skip.",
    )
    parser.add_argument(
        "--save-spectrograms",
        action="store_true",
        help="Save per-track spectrogram PNGs with Saraga annotations and estimated F0 overlaid.",
    )
    parser.add_argument(
        "--spectrogram-output-dir",
        default=DEFAULT_SPECTROGRAM_DIR,
        help="Directory for --save-spectrograms PNG files.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Attempt to download Saraga Hindustani through mirdata before local evaluation.",
    )
    parser.add_argument(
        "--partial-download",
        default=None,
        help="Comma-separated mirdata remote keys to download. Saraga Hindustani usually exposes 'all'.",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite existing files during mirdata download.",
    )
    return parser


if __name__ == "__main__":
    run_evaluation(build_arg_parser().parse_args())
