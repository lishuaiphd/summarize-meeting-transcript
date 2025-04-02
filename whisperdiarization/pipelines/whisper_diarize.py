from typing import List, Optional, Union

import numpy as np
import torch
from pyannote.audio import Pipeline
from torchaudio import functional as F
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers.pipelines.audio_utils import ffmpeg_read

class ASRDiarizationPipeline:

    def __init__(
        self,
        asr_pipeline,
        diarization_pipeline,
    ):
        self.asr_pipeline = asr_pipeline
        self.sampling_rate = asr_pipeline.feature_extractor.sampling_rate

        self.diarization_pipeline = diarization_pipeline

    @classmethod
    def from_pretrained(
        cls,
        asr_model: Optional[str] = "openai/whisper-large-v3",
        *,
        diarizer_model: Optional[str] = "pyannote/speaker-diarization",
        use_auth_token: Optional[Union[str, bool]] = False,
        **kwargs,
    ):
        # ASR pipeline
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.mps.is_available():
            device = "mps"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            asr_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(asr_model)
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            **kwargs,
        )

        # Diarization pipeline
        diarization_pipeline = Pipeline.from_pretrained(diarizer_model, use_auth_token=use_auth_token)

        return cls(asr_pipeline, diarization_pipeline)

    def __call__(
        self,
        inputs: Union[np.ndarray, List[np.ndarray]],
        group_by_speaker: bool = True,
        **kwargs,
    ):
        kwargs_asr = {
            argument[len("asr_"):]: value
            for argument, value in kwargs.items() if argument.startswith("asr_")
        }

        kwargs_diarization = {
            argument[len("diarization_"):]: value
            for argument, value in kwargs.items() if argument.startswith("diarization_")
        }

        # Preprocess audio
        _, diarizer_inputs = self.preprocess(inputs)

        # Diarization
        diarization = self.diarization_pipeline(
            {
                "waveform": diarizer_inputs,
                "sample_rate": self.sampling_rate
            },
            **kwargs_diarization,
        )
        diarization_segments = self.combine_diarization_segments(self.diarization_segments(diarization))

        # ASR transcript
        asr_out = self.asr_pipeline(
            inputs,
            return_timestamps=True,
            **kwargs_asr
        )
        asr_transcript = self.adjust_timestamps(self.remove_empty_text(asr_out["chunks"]))
        asr_end_timestamps = np.array([chunk["timestamp"][-1] for chunk in asr_transcript])

        # Align diarization with transcript
        dialog_segments = self.align_segments(diarization_segments, asr_transcript, asr_end_timestamps, group_by_speaker)

        return dialog_segments

    def preprocess(self, inputs):
        if isinstance(inputs, str):
            with open(inputs, "rb") as f:
                inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.sampling_rate)

        if isinstance(inputs, dict):
            if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                raise ValueError(
                    "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                    '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array")

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sampling_rate")
            inputs = _inputs
            if in_sampling_rate != self.sampling_rate:
                inputs = F.resample(torch.from_numpy(inputs), in_sampling_rate, self.sampling_rate).numpy()

        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for ASRDiarizePipeline")

        diarizer_inputs = torch.from_numpy(inputs).float()
        diarizer_inputs = diarizer_inputs.unsqueeze(0)

        return inputs, diarizer_inputs

    def remove_empty_text(self, chunks):
        adjusted_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk["text"]:
                new_start, new_end = chunk["timestamp"]
                new_chunk = {
                    "timestamp": (new_start, new_end),
                    "text": chunk["text"]
                }
                adjusted_chunks.append(new_chunk)
        return adjusted_chunks

    def adjust_timestamps(self, chunks):
        adjusted_chunks = []
        cumulative_offset = 0
        for i, chunk in enumerate(chunks):
            new_start, new_end = chunk["timestamp"]
            if i > 0:
                prev_start, prev_end = chunks[i - 1]["timestamp"]
                if (new_start + cumulative_offset) < (prev_start + cumulative_offset):
                    cumulative_offset += 30.0
                if new_end < new_start:
                    new_end += 30.0
            new_chunk = {
                "timestamp": (new_start + cumulative_offset, new_end + cumulative_offset),
                "text": chunk["text"]
            }
            adjusted_chunks.append(new_chunk)
        return adjusted_chunks

    def diarization_segments(self, diarization):
        segments = []
        for segment, track, label in diarization.itertracks(yield_label=True):
            segments.append({
                'segment': {
                    'start': segment.start,
                    'end': segment.end
                },
                'track': track,
                'label': label
            })
        return segments

    def combine_diarization_segments(self, segments):
        new_segments = []
        prev_segment = cur_segment = segments[0]
        for i in range(1, len(segments)):
            cur_segment = segments[i]
            if cur_segment["label"] != prev_segment["label"] and i < len(segments):
                new_segments.append({
                    "segment": {
                        "start": prev_segment["segment"]["start"],
                        "end": cur_segment["segment"]["start"]
                    },
                    "speaker": prev_segment["label"],
                })
                prev_segment = segments[i]

        new_segments.append({
            "segment": {
                "start": prev_segment["segment"]["start"],
                "end": cur_segment["segment"]["end"]
            },
            "speaker": prev_segment["label"],
        })

        return new_segments

    def align_segments(self, diarization_segments, asr_transcript, asr_end_timestamps, group_by_speaker=True):
        dialog_segments = []
        for segment in diarization_segments:
            # Get the diarizer end timestamp
            end_time = segment["segment"]["end"]

            if asr_end_timestamps.size == 0:
                continue

            # Find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
            upto_idx = np.argmin(np.abs(asr_end_timestamps - end_time))

            if group_by_speaker:
                dialog_segments.append({
                    "speaker":
                        segment["speaker"],
                    "text":
                        "".join([chunk["text"] for chunk in asr_transcript[:upto_idx + 1]]),
                    "timestamp": (asr_transcript[0]["timestamp"][0], asr_transcript[upto_idx]["timestamp"][1]),
                })
            else:
                for i in range(upto_idx + 1):
                    dialog_segments.append({"speaker": segment["speaker"], **asr_transcript[i]})

            asr_transcript = asr_transcript[upto_idx + 1:]
            asr_end_timestamps = asr_end_timestamps[upto_idx + 1:]
        return dialog_segments