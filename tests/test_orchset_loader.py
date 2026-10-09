from pathlib import Path

import pytest

import saraga


def test_load_orchset_tracks_and_annotations():
    dataset_root = Path("orchset")
    if not dataset_root.exists():
        pytest.skip("ORCHSET dataset is local-only; place it under orchset/ to run this smoke test.")

    tracks = saraga.load_orchset_tracks(dataset_root)
    if not tracks:
        pytest.skip("No ORCHSET tracks found under orchset/.")

    first_track = tracks[0]
    assert first_track.audio_path.exists()
    assert first_track.pitch_path.exists()

    times, frequencies = saraga.load_pitch_annotation(first_track, max_seconds=None)
    assert len(times) > 0
    assert len(frequencies) == len(times)
