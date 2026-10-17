
"""
OmniAI - Speaker Consistency Evaluation
File: backend/app/voice/speaker_consistency.py

Dependencies:
    pip install numpy scipy soundfile

Optional neural speaker embeddings:
    pip install torch torchaudio speechbrain

Features:
    - Speaker identity consistency evaluation
    - Reference vs generated voice comparison
    - ECAPA-TDNN speaker embeddings
    - Cosine similarity
    - Multiple audio segments
    - Speaker drift detection
    - Cross-speaker confusion analysis
    - Reference enrollment
    - Multi-speaker evaluation
    - Confidence thresholds
    - Batch and async evaluation
    - JSON report export

Security:
    Speaker embeddings are biometric data.
    Protect stored reference audio and embeddings.

Limitations:
    Speaker similarity is not proof of identity.
    Thresholds require calibration.
    This is not anti-spoofing or consent verification.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


EPS = 1e-12


class SpeakerConsistencyStatus(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class SpeakerConsistencyConfig:
    sample_rate: int = 16000

    embedding_model: str = (
        "speechbrain/spkrec-ecapa-voxceleb"
    )

    device: str = "cpu"

    min_audio_seconds: float = 1.0
    max_audio_seconds: float = 180.0

    similarity_threshold: float = 0.75
    drift_threshold: float = 0.15
    confusion_margin: float = 0.05

    excellent_threshold: float = 0.90
    good_threshold: float = 0.80
    fair_threshold: float = 0.65

    min_segments: int = 2
    max_history: int = 500

    output_dir: str = (
        "storage/evaluation/speaker_consistency"
    )

    def validate(self) -> None:
        if self.sample_rate != 16000:
            raise ValueError(
                "This ECAPA configuration requires "
                "16000 Hz audio"
            )

        if not (
            0 < self.min_audio_seconds
            <= self.max_audio_seconds
        ):
            raise ValueError(
                "Invalid audio duration limits"
            )

        for name in (
            "similarity_threshold",
            "drift_threshold",
            "confusion_margin",
            "excellent_threshold",
            "good_threshold",
            "fair_threshold",
        ):
            value = getattr(self, name)

            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        if not (
            self.excellent_threshold
            >= self.good_threshold
            >= self.fair_threshold
        ):
            raise ValueError(
                "Invalid score thresholds"
            )

        if self.min_segments < 1:
            raise ValueError(
                "min_segments must be >= 1"
            )

        if self.max_history < 1:
            raise ValueError(
                "max_history must be >= 1"
            )


@dataclass
class SpeakerReference:
    speaker_id: str
    reference_paths: list[str]
    embedding: np.ndarray
    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )


@dataclass
class SpeakerSegment:
    segment_id: str
    speaker_id: str
    audio_path: str
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if not self.segment_id:
            raise ValueError(
                "segment_id is required"
            )

        if not self.speaker_id:
            raise ValueError(
                "speaker_id is required"
            )

        if not self.audio_path:
            raise ValueError(
                "audio_path is required"
            )

        if (
            self.start_seconds is not None
            and self.start_seconds < 0
        ):
            raise ValueError(
                "start_seconds cannot be negative"
            )

        if (
            self.end_seconds is not None
            and self.end_seconds < 0
        ):
            raise ValueError(
                "end_seconds cannot be negative"
            )

        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds
            <= self.start_seconds
        ):
            raise ValueError(
                "end_seconds must exceed start_seconds"
            )


@dataclass
class SegmentConsistencyResult:
    segment_id: str
    speaker_id: str
    audio_path: str
    reference_similarity: float
    consistency_score: float
    matches_reference: bool
    closest_speaker_id: Optional[str]
    closest_speaker_similarity: Optional[float]
    speaker_confusion: bool
    drift_from_previous: Optional[float]
    warnings: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpeakerMetrics:
    speaker_id: str
    segment_count: int
    mean_similarity: float
    minimum_similarity: float
    maximum_similarity: float
    similarity_std: float
    reference_match_rate: float
    temporal_consistency: float
    drift_count: int
    confusion_count: int
    overall_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpeakerConsistencyResult:
    evaluation_id: str
    status: SpeakerConsistencyStatus
    speaker_count: int
    segment_count: int
    overall_score: Optional[float]
    speaker_metrics: list[SpeakerMetrics]
    segment_results: list[SegmentConsistencyResult]
    warnings: list[str] = field(
        default_factory=list
    )
    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "status": self.status.value,
            "speaker_count": self.speaker_count,
            "segment_count": self.segment_count,
            "overall_score": self.overall_score,
            "speaker_metrics": [
                item.to_dict()
                for item in self.speaker_metrics
            ],
            "segment_results": [
                item.to_dict()
                for item in self.segment_results
            ],
            "warnings": self.warnings,
            "created_at": self.created_at,
        }


class ECAPASpeakerEmbeddingBackend:
    """
    SpeechBrain ECAPA-TDNN speaker embedding backend.

    Models are loaded lazily to avoid startup
    overhead when speaker evaluation is unused.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        cache_dir: str,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            try:
                from speechbrain.inference.speaker import (
                    EncoderClassifier,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Install speechbrain and torch "
                    "to enable speaker embeddings"
                ) from exc

            self._model = (
                EncoderClassifier.from_hparams(
                    source=self.model_name,
                    savedir=self.cache_dir,
                    run_opts={
                        "device": self.device
                    },
                )
            )

            return self._model

    def extract(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required"
            ) from exc

        model = self._load_model()

        waveform = torch.from_numpy(
            audio.astype(np.float32)
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            embedding = model.encode_batch(
                waveform
            )

        vector = (
            embedding.detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32)
        )

        norm = float(
            np.linalg.norm(vector)
        )

        if norm <= EPS:
            raise ValueError(
                "Invalid speaker embedding"
            )

        return vector / norm


class SpeakerConsistencyEvaluator:
    """
    Evaluates voice identity consistency
    across reference recordings and segments.

    A speaker must be explicitly enrolled
    before their segments can be evaluated.
    """

    def __init__(
        self,
        config: Optional[
            SpeakerConsistencyConfig
        ] = None,
        backend: Optional[
            ECAPASpeakerEmbeddingBackend
        ] = None,
    ):
        self.config = (
            config or SpeakerConsistencyConfig()
        )
        self.config.validate()

        self.output_dir = Path(
            self.config.output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.backend = backend or (
            ECAPASpeakerEmbeddingBackend(
                model_name=(
                    self.config.embedding_model
                ),
                device=self.config.device,
                cache_dir=str(
                    self.output_dir / "model_cache"
                ),
            )
        )

        self.references: dict[
            str, SpeakerReference
        ] = {}

        self.history: list[
            SpeakerConsistencyResult
        ] = []

    @staticmethod
    def _cosine_similarity(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        denominator = (
            np.linalg.norm(first)
            * np.linalg.norm(second)
        )

        if denominator <= EPS:
            return 0.0

        similarity = float(
            np.dot(first, second)
            / denominator
        )

        return float(
            np.clip(
                similarity,
                -1.0,
                1.0,
            )
        )

    @staticmethod
    def _similarity_score(
        cosine_similarity: float,
    ) -> float:
        """
        Convert cosine [-1, 1] to [0, 1].

        This is not a calibrated probability
        of speaker identity.
        """
        return float(
            np.clip(
                (cosine_similarity + 1.0) / 2.0,
                0.0,
                1.0,
            )
        )

    def _load_audio(
        self,
        audio_path: str,
        start_seconds: Optional[float] = None,
        end_seconds: Optional[float] = None,
    ) -> np.ndarray:
        path = Path(audio_path)

        if not path.is_file():
            raise FileNotFoundError(
                audio_path
            )

        info = sf.info(str(path))

        start = (
            0.0
            if start_seconds is None
            else start_seconds
        )

        end = (
            info.duration
            if end_seconds is None
            else end_seconds
        )

        if (
            start < 0
            or end <= start
            or end > info.duration + 1e-6
        ):
            raise ValueError(
                "Invalid segment boundaries"
            )

        duration = end - start

        if not (
            self.config.min_audio_seconds
            <= duration
            <= self.config.max_audio_seconds
        ):
            raise ValueError(
                "Segment duration outside allowed range"
            )

        with sf.SoundFile(str(path)) as handle:
            start_frame = int(
                start * handle.samplerate
            )

            end_frame = min(
                len(handle),
                int(
                    end * handle.samplerate
                ),
            )

            handle.seek(start_frame)

            audio = handle.read(
                frames=end_frame - start_frame,
                dtype="float32",
                always_2d=True,
            )

            sample_rate = handle.samplerate

        if audio.size == 0:
            raise ValueError(
                "Audio segment is empty"
            )

        if not np.isfinite(audio).all():
            raise ValueError(
                "Invalid audio samples"
            )

        audio = audio.mean(axis=1)

        if sample_rate != self.config.sample_rate:
            divisor = math.gcd(
                sample_rate,
                self.config.sample_rate,
            )

            audio = resample_poly(
                audio,
                self.config.sample_rate // divisor,
                sample_rate // divisor,
            ).astype(np.float32)

        if (
            float(np.sqrt(
                np.mean(
                    audio.astype(np.float64) ** 2
                )
            ))
            < 1e-5
        ):
            raise ValueError(
                "Audio contains insufficient signal"
            )

        return audio

    def _embedding(
        self,
        audio_path: str,
        start_seconds: Optional[float] = None,
        end_seconds: Optional[float] = None,
    ) -> np.ndarray:
        audio = self._load_audio(
            audio_path,
            start_seconds,
            end_seconds,
        )

        return self.backend.extract(
            audio
        )

    def enroll_speaker(
        self,
        speaker_id: str,
        reference_paths: Sequence[str],
    ) -> SpeakerReference:
        if not speaker_id.strip():
            raise ValueError(
                "speaker_id is required"
            )

        paths = list(reference_paths)

        if not paths:
            raise ValueError(
                "At least one reference recording "
                "is required"
            )

        embeddings = [
            self._embedding(path)
            for path in paths
        ]

        mean_embedding = np.mean(
            np.stack(embeddings),
            axis=0,
        )

        norm = float(
            np.linalg.norm(mean_embedding)
        )

        if norm <= EPS:
            raise ValueError(
                "Invalid reference embedding"
            )

        mean_embedding = (
            mean_embedding / norm
        ).astype(np.float32)

        reference = SpeakerReference(
            speaker_id=speaker_id,
            reference_paths=paths,
            embedding=mean_embedding,
        )

        self.references[
            speaker_id
        ] = reference

        return reference

    async def enroll_speaker_async(
        self,
        speaker_id: str,
        reference_paths: Sequence[str],
    ) -> SpeakerReference:
        return await asyncio.to_thread(
            self.enroll_speaker,
            speaker_id,
            reference_paths,
        )

    def remove_speaker(
        self,
        speaker_id: str,
    ) -> bool:
        return (
            self.references.pop(
                speaker_id,
                None,
            )
            is not None
        )

    def _closest_speaker(
        self,
        embedding: np.ndarray,
    ) -> tuple[
        Optional[str],
        Optional[float],
    ]:
        if not self.references:
            return None, None

        similarities = {
            speaker_id: self._cosine_similarity(
                embedding,
                reference.embedding,
            )
            for speaker_id, reference
            in self.references.items()
        }

        speaker_id = max(
            similarities,
            key=similarities.get,
        )

        return (
            speaker_id,
            similarities[speaker_id],
        )

    def _evaluate_segment(
        self,
        segment: SpeakerSegment,
        previous_embedding: Optional[np.ndarray],
    ) -> tuple[
        SegmentConsistencyResult,
        np.ndarray,
    ]:
        segment.validate()

        reference = self.references.get(
            segment.speaker_id
        )

        if reference is None:
            raise ValueError(
                "Speaker is not enrolled: "
                f"{segment.speaker_id}"
            )

        embedding = self._embedding(
            segment.audio_path,
            segment.start_seconds,
            segment.end_seconds,
        )

        similarity = self._cosine_similarity(
            embedding,
            reference.embedding,
        )

        closest_id, closest_similarity = (
            self._closest_speaker(
                embedding
            )
        )

        confusion = False

        if (
            closest_id is not None
            and closest_id != segment.speaker_id
            and closest_similarity is not None
        ):
            confusion = (
                closest_similarity
                >= similarity
                + self.config.confusion_margin
            )

        drift = None

        if previous_embedding is not None:
            drift = (
                1.0
                - self._cosine_similarity(
                    previous_embedding,
                    embedding,
                )
            )

        warnings = []

        if (
            similarity
            < self.config.similarity_threshold
        ):
            warnings.append(
                "Speaker similarity below threshold"
            )

        if (
            drift is not None
            and drift > self.config.drift_threshold
        ):
            warnings.append(
                "Possible speaker embedding drift"
            )

        if confusion:
            warnings.append(
                "Another enrolled speaker has "
                "a substantially higher similarity"
            )

        result = SegmentConsistencyResult(
            segment_id=segment.segment_id,
            speaker_id=segment.speaker_id,
            audio_path=segment.audio_path,
            reference_similarity=round(
                similarity, 4
            ),
            consistency_score=round(
                self._similarity_score(
                    similarity
                ),
                4,
            ),
            matches_reference=(
                similarity
                >= self.config.similarity_threshold
            ),
            closest_speaker_id=closest_id,
            closest_speaker_similarity=(
                round(closest_similarity, 4)
                if closest_similarity is not None
                else None
            ),
            speaker_confusion=confusion,
            drift_from_previous=(
                round(drift, 4)
                if drift is not None
                else None
            ),
            warnings=warnings,
        )

        return result, embedding

    def _speaker_metrics(
        self,
        speaker_id: str,
        results: Sequence[
            SegmentConsistencyResult
        ],
    ) -> SpeakerMetrics:
        similarities = np.asarray(
            [
                result.reference_similarity
                for result in results
            ],
            dtype=np.float64,
        )

        matches = [
            result.matches_reference
            for result in results
        ]

        drifts = [
            result.drift_from_previous
            for result in results
            if result.drift_from_previous is not None
        ]

        drift_count = sum(
            drift > self.config.drift_threshold
            for drift in drifts
        )

        confusion_count = sum(
            result.speaker_confusion
            for result in results
        )

        mean_similarity = float(
            np.mean(similarities)
        )

        match_rate = float(
            np.mean(matches)
        )

        temporal_consistency = (
            float(
                np.mean(
                    [
                        max(
                            0.0,
                            1.0 - drift,
                        )
                        for drift in drifts
                    ]
                )
            )
            if drifts
            else 1.0
        )

        overall = (
            0.50
            * self._similarity_score(
                mean_similarity
            )
            + 0.30 * match_rate
            + 0.20 * temporal_consistency
        )

        if results:
            overall *= (
                1.0
                - 0.20
                * confusion_count
                / len(results)
            )

        overall = float(
            np.clip(
                overall,
                0.0,
                1.0,
            )
        )

        return SpeakerMetrics(
            speaker_id=speaker_id,
            segment_count=len(results),
            mean_similarity=round(
                mean_similarity, 4
            ),
            minimum_similarity=round(
                float(np.min(similarities)),
                4,
            ),
            maximum_similarity=round(
                float(np.max(similarities)),
                4,
            ),
            similarity_std=round(
                float(np.std(similarities)),
                4,
            ),
            reference_match_rate=round(
                match_rate, 4
            ),
            temporal_consistency=round(
                temporal_consistency, 4
            ),
            drift_count=drift_count,
            confusion_count=confusion_count,
            overall_score=round(
                overall, 4
            ),
        )

    def _status(
        self,
        score: Optional[float],
        segment_count: int,
    ) -> SpeakerConsistencyStatus:
        if (
            score is None
            or segment_count < self.config.min_segments
        ):
            return (
                SpeakerConsistencyStatus.INSUFFICIENT_DATA
            )

        if score >= self.config.excellent_threshold:
            return SpeakerConsistencyStatus.EXCELLENT

        if score >= self.config.good_threshold:
            return SpeakerConsistencyStatus.GOOD

        if score >= self.config.fair_threshold:
            return SpeakerConsistencyStatus.FAIR

        return SpeakerConsistencyStatus.POOR

    def evaluate(
        self,
        segments: Sequence[SpeakerSegment],
    ) -> SpeakerConsistencyResult:
        segments = list(segments)

        segment_ids = [
            segment.segment_id
            for segment in segments
        ]

        if (
            len(segment_ids)
            != len(set(segment_ids))
        ):
            raise ValueError(
                "Duplicate segment IDs"
            )

        for segment in segments:
            segment.validate()

        grouped: dict[
            str, list[SpeakerSegment]
        ] = {}

        for segment in segments:
            grouped.setdefault(
                segment.speaker_id,
                [],
            ).append(segment)

        segment_results = []
        speaker_metrics = []
        warnings = []

        for speaker_id, speaker_segments in (
            sorted(grouped.items())
        ):
            previous_embedding = None
            current_results = []

            for segment in speaker_segments:
                result, embedding = (
                    self._evaluate_segment(
                        segment,
                        previous_embedding,
                    )
                )

                current_results.append(result)
                segment_results.append(result)

                previous_embedding = embedding

            speaker_metrics.append(
                self._speaker_metrics(
                    speaker_id,
                    current_results,
                )
            )

        if speaker_metrics:
            total_segments = sum(
                metric.segment_count
                for metric in speaker_metrics
            )

            overall = sum(
                metric.overall_score
                * metric.segment_count
                for metric in speaker_metrics
            ) / total_segments

        else:
            overall = None

        if (
            len(segments)
            < self.config.min_segments
        ):
            warnings.append(
                "Insufficient segments for "
                "reliable consistency evaluation"
            )

        warnings.append(
            "Scores are heuristic and require "
            "speaker-model threshold calibration."
        )

        result = SpeakerConsistencyResult(
            evaluation_id=(
                f"speaker_eval_{uuid.uuid4().hex}"
            ),
            status=self._status(
                overall,
                len(segments),
            ),
            speaker_count=len(grouped),
            segment_count=len(segments),
            overall_score=(
                round(overall, 4)
                if overall is not None
                else None
            ),
            speaker_metrics=speaker_metrics,
            segment_results=segment_results,
            warnings=warnings,
        )

        self.history.append(result)

        if (
            len(self.history)
            > self.config.max_history
        ):
            self.history = self.history[
                -self.config.max_history:
            ]

        return result

    async def evaluate_async(
        self,
        segments: Sequence[SpeakerSegment],
    ) -> SpeakerConsistencyResult:
        return await asyncio.to_thread(
            self.evaluate,
            segments,
        )

    async def evaluate_batch(
        self,
        batches: Sequence[
            Sequence[SpeakerSegment]
        ],
        max_concurrency: int = 2,
    ) -> list[SpeakerConsistencyResult]:
        if max_concurrency < 1:
            raise ValueError(
                "max_concurrency must be >= 1"
            )

        semaphore = asyncio.Semaphore(
            max_concurrency
        )

        async def worker(
            batch: Sequence[SpeakerSegment],
        ):
            async with semaphore:
                return await (
                    self.evaluate_async(batch)
                )

        return await asyncio.gather(
            *[
                worker(batch)
                for batch in batches
            ]
        )

    def export_json(
        self,
        result: SpeakerConsistencyResult,
    ) -> Path:
        path = (
            self.output_dir
            / f"{result.evaluation_id}.json"
        )

        path.write_text(
            json.dumps(
                result.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return path

    def get_history(
        self,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        results = self.history

        if limit is not None:
            if limit < 0:
                raise ValueError(
                    "limit must be >= 0"
                )

            results = (
                results[-limit:]
                if limit
                else []
            )

        return [
            result.to_dict()
            for result in results
        ]


async def main():
    evaluator = SpeakerConsistencyEvaluator()

    reference_path = "reference_voice.wav"

    segment_paths = [
        "generated_voice_1.wav",
        "generated_voice_2.wav",
    ]

    required_paths = [
        reference_path,
        *segment_paths,
    ]

    if not all(
        Path(path).exists()
        for path in required_paths
    ):
        print(
            "Provide reference_voice.wav and "
            "generated_voice_1.wav / "
            "generated_voice_2.wav"
        )
        return

    await evaluator.enroll_speaker_async(
        speaker_id="assistant",
        reference_paths=[
            reference_path
        ],
    )

    segments = [
        SpeakerSegment(
            segment_id=f"segment_{index}",
            speaker_id="assistant",
            audio_path=path,
        )
        for index, path in enumerate(
            segment_paths,
            start=1,
        )
    ]

    result = await evaluator.evaluate_async(
        segments
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )

    print(
        "Report:",
        evaluator.export_json(result),
    )


if __name__ == "__main__":
    asyncio.run(main())
