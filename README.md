# SpeechT5 ASR Attack

This repository contains implementations of adversarial attacks (PGD and Carlini & Wagner) against the Microsoft SpeechT5 ASR model. The code produces adversarial audio that attempts to force the ASR model to output a specified target transcription while maintaining a minimum audio quality (SNR).

## Requirements
- Python 3.8+
- PyTorch
- transformers
- librosa
- soundfile
- numpy
- huggingface_hub
- fsspec

Install required packages (example):
```bash
pip install -r requirements.txt
```

## Quick Usage

Example command:
```bash
python main.py --audio_path /kaggle/working/demo.mp3 \
               --target_text "attack on model was successful" \
               --output_path "/kaggle/working/SpeechT5-AST-Attack/adversarial_audio" \
               --snr 20.0 \
               --num_iter 2000 \
               --attack_type "all"
```

## Parameters (main.py)
- --audio_path (str, required): Path to input audio file.
- --target_text (str, required): Desired transcription for adversarial audio.
- --output_path (str, required): Base path for saving output audio (suffixes `_pgd.wav` or `_cw.wav` are appended).
- --snr (float, default=20.0): Minimum acceptable SNR (dB) for adversarial audio.
- --num_iter (int, default=500): Number of iterations for the chosen attack.
- --attack_type (str, default=`pgd`): Choose `pgd`, `cw`, or `all`.


## How it works
- PGD: iteratively updates a bounded perturbation using gradient steps to increase loss toward target transcription, projecting back into epsilon-ball.
- C&W: treats adversarial perturbation as an optimization variable in unconstrained space (tanh trick), minimizes total loss = c * attack_loss + perturbation_loss, adjusts c via binary search for smallest perturbation that achieves the target while meeting SNR constraint.


## Output
- Saved audio files appended with `_pgd.wav` or `_cw.wav`.

## Testing Adversarial Audio
Use `test.py` to verify the transcription of generated adversarial audio files:

```python
python test.py
```