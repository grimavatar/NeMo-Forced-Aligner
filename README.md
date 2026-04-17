# NeMo Forced Aligner (NFA)

<p align="center">
Try it out: <a href="https://huggingface.co/spaces/erastorgueva-nv/NeMo-Forced-Aligner">HuggingFace Space 🎤</a> | Tutorial: <a href="https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/tools/NeMo_Forced_Aligner_Tutorial.ipynb">"How to use NFA?" 🚀</a> | Blog post: <a href="https://nvidia.github.io/NeMo/blogs/2023/2023-08-forced-alignment/">"How does forced alignment work?" 📚</a>
</p>

<p align="center">
<img width="80%" src="https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/nfa_forced_alignment_pipeline.png">
</p>

NFA is a tool for generating token-, word- and segment-level timestamps of speech in audio using NeMo's CTC-based Automatic Speech Recognition models. You can provide your own reference text, or use ASR-generated transcription. You can use NeMo's ASR Model checkpoints out of the box in [14+ languages](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/results.html#speech-recognition-languages), or train your own model. NFA can be used on long audio files of 1+ hours duration (subject to your hardware and the ASR model used).


## Install

```bash
pip install git+https://github.com/grimavatar/NeMo-Forced-Aligner.git
```

## Example

```python
from pathlib import Path
from forced_aligner import ForcedAligner

model_name = "nvidia/parakeet-tdt_ctc-1.1b"  			# Top 1 (1.1b) - Best
# model_name = "stt_en_fastconformer_ctc_xxlarge"  		# Top 2 (1.1b)
# model_name = "stt_en_fastconformer_ctc_xlarge"  		# Top 3 (0.6b)
# model_name = "stt_en_fastconformer_hybrid_large_pc"   # Top 4 (110m) - Default

aligner = ForcedAligner(pretrained_name = model_name)

audio_paths = [str(e) for e in Path(".").absolute().glob("*wav")]
text_paths = [str(Path(e).with_suffix(".txt")) for e in audio_paths]

utt_data = aligner.align(audio_path, text_path)
alignment = aligner.simplify(utt_data)
```
