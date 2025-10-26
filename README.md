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
               --output_path "/kaggle/working/SpeechT5-AST-Attack/adversarial_audio1_20dB" \
               --snr 15.0 \
               --num_iter 2000 \
               --attack_type "all"
```

## Parameters (CLI / main.py)
- --audio_path (str, required): Path to input audio file.
- --target_text (str, required): Desired transcription for adversarial audio.
- --output_path (str, required): Base path for saving output audio (suffixes `_pgd.wav` or `_cw.wav` are appended).
- --snr (float, default=20.0): Minimum acceptable SNR (dB) for adversarial audio.
- --num_iter (int, default=500): Number of iterations for the chosen attack.
- --attack_type (str, default=`pgd`): Choose `pgd`, `cw`, or `all`.

## Attack parameter summaries (in-code defaults)
- PGD:
  - epsilon: maximum perturbation per sample (default in code: 0.2)
  - alpha: step size per iteration (default in code: 0.01)
  - num_iterations: number of PGD steps
  - early_stop: stop if model already predicts target

- C&W:
  - c / initial_const: balance between attack success and perturbation size (binary searched)
  - learning_rate: optimizer LR (default in code: 0.01)
  - num_iterations: optimization steps per binary-search step
  - binary_search_steps: number of constants to test
  - min_snr_db: desired minimum SNR

## How it works (brief)
- PGD: iteratively updates a bounded perturbation using gradient steps to increase loss toward target transcription, projecting back into epsilon-ball.
- C&W: treats adversarial perturbation as an optimization variable in unconstrained space (tanh trick), minimizes total loss = c * attack_loss + perturbation_loss, adjusts c via binary search for smallest perturbation that achieves the target while meeting SNR constraint.

## Files
- main.py — CLI entrypoint and wrappers for both attacks.
- pgd_attack.py — PGD attack implementation (expected in same folder).
- cw_attack.py — C&W attack implementation (contains CWAttackASR class).
- README.md — this file.

## Output
- Saved audio files appended with `_pgd.wav` or `_cw.wav`.
- main functions return (adversarial_audio_numpy, attack_info) where `attack_info` contains:
  - original_transcription, target_transcription, final_transcription
  - snr_db, attack_successful, best_const (C&W), binary_search_steps, iterations info
