# Melody - IKS

Melody - IKS is a research codebase for fundamental-frequency extraction and
tonic-relative swara mapping in vocal Indian Classical Music (ICM). It supports
the conference paper **"A Multi-Paradigm Comparative Study on Fundamental
Frequency Extraction and Swara Mapping in Indian Classical Music"** by Keshav
Raj.

The project compares pitch-tracking paradigms that are useful for Hindustani
music, where continuous glides, gamakas, tanpura resonance, and performer-
specific tonic choices make Western fixed-pitch assumptions brittle.

## Paper Overview

The paper evaluates five computational approaches on the Saraga Hindustani
dataset:

- Traditional machine learning with Random Forest spectral-feature
  classification
- Probabilistic signal processing with YIN / pYIN-style pitch tracking
- Spectral harmonic-template matching with SWIPE
- Harmonic-salience / CRNN-style pitch representation
- Weighted-median multi-model consensus ensembling

The pipeline frames audio at 16 kHz with a 2,048-sample analysis window and a
160-sample hop, estimates continuous f0 contours, and maps each voiced frame to
Indian Classical Music swaras relative to the performer-specific tonic Sa:

```text
n(k) = 12 * log2(f0(k) / Sa)
swara = round(n(k)) mod 12
```

## Reported Results

The paper reports MIREX-style pitch metrics with a 50-cent tolerance over 445
method-track instances from Saraga Hindustani.

| Model paradigm | RPA (%) | RCA (%) | VR (%) | VFA (%) | OA (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Traditional ML (RF) | 54.80 | 64.94 | 69.57 | 44.47 | 55.14 |
| DSP / Probabilistic (YIN) | 59.49 | 81.60 | 97.60 | 55.07 | 51.54 |
| Harmonic Template (SWIPE) | 66.03 | 72.64 | 83.21 | 60.12 | 54.88 |
| CRNN / Salience Representation | 73.14 | 79.61 | 90.55 | 46.76 | **62.59** |
| Consensus Ensemble | **74.05** | **85.05** | **99.50** | **86.72** | 50.31 |

The weighted consensus ensemble gives the strongest Raw Pitch Accuracy and Raw
Chroma Accuracy, while the harmonic-salience / CRNN representation gives the
highest Overall Accuracy because of stronger voicing discrimination. These
values are preserved in `paper_scores.csv`.

## Repository Layout

- `saraga.py` - dataset loading, pitch estimators, evaluation metrics,
  spectrogram plotting, and validation CLI
- `New.py` - compact implementation of the five-paradigm extraction and swara
  mapping pipeline described in the paper
- `saraga_colab.ipynb` - notebook workflow for interactive experiments
- `paper_scores.csv` - paper-level benchmark table
- `saraga_hindustani_validation_results.csv` - track-level validation output
  from the local reproducibility run
- `*_validation_graph.png` - generated comparison figures
- `tests/` - smoke tests for local dataset loaders

Raw datasets, downloaded archives, virtual environments, caches, and generated
bulk spectrograms are intentionally ignored by git.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

CREPE support is optional and can be installed separately if you want to run the
`crepe` method.

## Dataset Setup

For Saraga Hindustani, place the dataset under one of these local paths:

```text
saraga_hindustani/
saraga1.5_hindustani/
```

The validation CLI can also download Saraga through `mirdata`:

```powershell
python saraga.py --download --dataset-type saraga
```

For ORCHSET loader tests, place the ORCHSET dataset under:

```text
orchset/
```

Dataset and recording licenses belong to their original providers. The paper
attributes Saraga Hindustani to the Music Technology Group (MTG), Universitat
Pompeu Fabra.

## Running

Show the available validation options:

```powershell
python saraga.py --help
```

Run a short Saraga evaluation:

```powershell
python saraga.py --dataset-type saraga --limit 3 --max-seconds 30 --methods ensemble,pyin,salience,swipe,yin
```

Run ORCHSET validation:

```powershell
python saraga.py --dataset-type orchset --methods ensemble,pyin,salience,swipe,yin
```

Run tests:

```powershell
pytest
```

Data-dependent tests skip automatically when the corresponding local dataset is
not present.

## Swara Mapping

The project maps frequency estimates to tonic-relative swaras using this pitch
class dictionary:

| Index | Swara |
| ---: | --- |
| 0 | Shuddha Sa (S) |
| 1 | Komal Re (r) |
| 2 | Shuddha Re (R) |
| 3 | Komal Ga (g) |
| 4 | Shuddha Ga (G) |
| 5 | Shuddha Ma (m) |
| 6 | Tivra Ma (M) |
| 7 | Shuddha Pa (P) |
| 8 | Komal Dha (d) |
| 9 | Shuddha Dha (D) |
| 10 | Komal Ni (n) |
| 11 | Shuddha Ni (N) |

## Research Direction

The next planned extensions are transformer-based pitch representations,
fine-tuned neural melody estimators for Indian Classical Music, and stronger
dynamic tonic tracking for long-form raga performance analysis.
