## Introduction
Whiper diarization performs a diarized transcript of audio meetings. The result is a transcript with timestamped
dialog and speaker names. This program can also summarize the transcript.

The solution is based on the Open-AI Whisper model for audio transcription. Diarization use the pyannote library.
Summaries are generated with the BART large CNN model.

## Installation

### Install requirements

```bash
pip install whisperdiarization transformers
pip install -r requirements/speaker_diarization.txt
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

### Accept licensing and permissions of two models

You must confirm the licensing permissions of these two models.

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

## Usage

### Python

```python
from whisperdiarization.pipelines.long_text_summarization import LongTextSummarizationPipeline
from whisperdiarization.pipelines.whisper_diarize import ASRDiarizationPipeline
from whisperdiarization import format_speech_to_dialogue

audio_path = "audio/audio.mp3"
pipeline = ASRDiarizationPipeline.from_pretrained(
    asr_model="openai/whisper-large-v3",
    diarizer_model="pyannote/speaker-diarization-3.1"
)

output_text = pipeline(audio_path, asr_generate_kwargs={"language": "english"})
dialogue = format_speech_to_dialogue(output_text)
print(dialogue)

summarizer = LongTextSummarizationPipeline(model_id="facebook/bart-large-cnn")
summary = summarizer.summarize(dialogue)
print(summary)
```
### CLI

Work in progress

### API

Work in progress