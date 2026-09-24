"""Load RAVDESS dataset with dual sample-rate support."""

import os
import re
import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import config


@dataclass
class RavdessItem:
    path: str
    actor: int
    emotion: int
    intensity: int
    statement: int
    gender: str  # "M" or "F"


def parse_filename(path: str) -> Optional[RavdessItem]:
    fname = os.path.basename(path)
    match = re.match(
        r"(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav",
        fname,
    )
    if not match:
        return None
    parts = [int(x) for x in match.groups()]
    modality, vocal, emotion, intensity, statement, repetition, actor = parts
    if modality != 3 or vocal != 1:
        return None
    gender = "M" if actor % 2 == 1 else "F"
    return RavdessItem(
        path=path,
        actor=actor,
        emotion=emotion,
        intensity=intensity,
        statement=statement,
        gender=gender,
    )


def discover_files(root: str = None) -> list[RavdessItem]:
    root = root or config.RAVDESS_ROOT
    items = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if not fname.endswith(".wav"):
                continue
            full = os.path.join(dirpath, fname)
            item = parse_filename(full)
            if item is not None:
                items.append(item)
    items.sort(key=lambda x: (x.actor, x.emotion, x.intensity, x.statement))
    return items


def load_audio(path: str, sr: int, duration: float = None) -> np.ndarray:
    duration = duration or config.DURATION_SEC
    y, _ = librosa.load(path, sr=sr, duration=duration)
    max_len = int(sr * duration)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]
    return y


class RavdessDataset:
    """In-memory RAVDESS dataset with preloaded waveforms at a given SR."""

    def __init__(self, items: list[RavdessItem] = None, sr: int = None,
                 root: str = None, load_audio_flag: bool = True):
        self.items = items if items is not None else discover_files(root)
        self.sr = sr or config.SR_HUBERT

        self.paths = np.array([it.path for it in self.items])
        self.actors = np.array([it.actor for it in self.items])
        self.labels = np.array([it.emotion - 1 for it in self.items])  # 0-indexed
        self.emotions = np.array([it.emotion for it in self.items])
        self.genders = np.array([it.gender for it in self.items])

        self.waveforms: Optional[list[np.ndarray]] = None
        if load_audio_flag:
            self._load_all()

    def _load_all(self):
        print(f"[INFO] Loading {len(self.items)} files at {self.sr} Hz ...")
        self.waveforms = []
        for item in self.items:
            y = load_audio(item.path, self.sr)
            self.waveforms.append(y)
        print(f"[INFO] Loaded {len(self.waveforms)} waveforms")

    def __len__(self):
        return len(self.items)

    def get_waveform(self, idx: int) -> np.ndarray:
        return self.waveforms[idx]

    def get_label(self, idx: int) -> int:
        return int(self.labels[idx])

    def class_distribution(self) -> dict[str, int]:
        dist = {}
        for emo_code in sorted(config.EMOTION_LABELS.keys()):
            name = config.EMOTION_LABELS[emo_code]
            count = int(np.sum(self.emotions == emo_code))
            dist[name] = count
        return dist

    def print_summary(self):
        print(f"[INFO] RAVDESS summary: {len(self)} files, "
              f"{len(np.unique(self.actors))} actors, SR={self.sr}")
        dist = self.class_distribution()
        for name, count in dist.items():
            print(f"  {name:12s}: {count}")
