import numpy as np
from pgd_attack import PGDAttackASR
from cw_attack import CWAttackASR
import torch
import argparse
import soundfile as sf
import librosa


def pgd_attack(audio_path, target_text, output_path, min_snr, num_iter, alpha):
    audio, sample_rate = load_audio(audio_path, target_sr=16000)
    
    attacker = PGDAttackASR(
        model_name="microsoft/speecht5_asr",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    adversarial_audio, attack_info = attacker.pgd_attack(
        audio=audio,
        target_text=target_text,
        sample_rate=sample_rate,
        alpha=alpha,
        num_iterations=num_iter,
        min_snr_db=min_snr,
        early_stop=True
    )
    
    output_file = output_path + "_pgd.wav"
    save_audio(adversarial_audio, output_file, sample_rate)
    print(f"Saved to: {output_file}\n")
    
    return adversarial_audio, attack_info


def cw_attack(audio_path, target_text, output_path, min_snr, num_iter, binary_steps):
    audio, sample_rate = load_audio(audio_path, target_sr=16000)
    
    attacker = CWAttackASR(
        model_name="microsoft/speecht5_asr",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    adversarial_audio, attack_info = attacker.cw_attack(
        audio=audio,
        target_text=target_text,
        sample_rate=sample_rate,
        min_snr_db=min_snr,
        num_iterations=num_iter,
        binary_search_steps=binary_steps
    )
    
    output_file = output_path + "_cw.wav"
    save_audio(adversarial_audio, output_file, sample_rate)
    print(f"Saved to: {output_file}\n")
    
    return adversarial_audio, attack_info


def load_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio, sr


def save_audio(audio, output_path, sample_rate=16000):
    sf.write(output_path, audio, sample_rate, subtype='PCM_16')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR adversarial attack")
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--target_text", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--binary_steps", type=int, default=5)
    parser.add_argument("--attack_type", default="pgd")
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args()

    if args.attack_type == "pgd":
        pgd_attack(args.audio_path, args.target_text, args.output_path, args.snr, args.num_iter, args.alpha)
    elif args.attack_type == "cw":
        cw_attack(args.audio_path, args.target_text, args.output_path, args.snr, args.num_iter, args.binary_steps)
    elif args.attack_type == "all":
        print("\n=== PGD Attack ===")
        pgd_attack(args.audio_path, args.target_text, args.output_path, args.snr, args.num_iter)
        print("\n=== C&W Attack ===")
        cw_attack(args.audio_path, args.target_text, args.output_path, args.snr, args.num_iter, args.binary_steps)
    else:
        print(f"Unknown attack type: {args.attack_type}")