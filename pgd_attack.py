import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import warnings

warnings.filterwarnings('ignore')


class PGDAttackASR:
    def __init__(self, model_name="microsoft/speecht5_asr", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = False
    
    def calculate_snr(self, original, adversarial):
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean((adversarial - original) ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * torch.log10(signal_power / noise_power)
        return snr.item()
    
    def calculate_epsilon(self, original, min_snr_db=20.0):
        signal_power = torch.mean(original ** 2)
        max_noise_power = signal_power / (10 ** (min_snr_db / 10))
        epsilon = torch.sqrt(max_noise_power)
        
        return epsilon
    
    def compute_loss(self, audio_values, target_ids):
        outputs = self.model(input_values=audio_values, labels=target_ids, return_dict=True)
        return outputs.loss
    
    def pgd_attack(self, audio, target_text, sample_rate=16000, alpha=0.02,
                   num_iterations=2000, min_snr_db=20.0, early_stop=True, 
                   alpha_decay_patience=20, alpha_decay_factor=0.9):
        
        inputs = self.processor(audio=audio, sampling_rate=sample_rate, return_tensors="pt")
        audio_values = inputs.input_values.to(self.device)
        
        target_inputs = self.processor.tokenizer(target_text, return_tensors="pt")
        target_ids = target_inputs.input_ids.to(self.device)
        
        adv_audio = audio_values.clone().detach()
        original_audio = audio_values.clone().detach()
        
        with torch.no_grad():
            original_outputs = self.model.generate(audio_values)
            original_transcription = self.processor.tokenizer.decode(original_outputs[0], skip_special_tokens=True)

        epsilon = self.calculate_epsilon(original_audio, min_snr_db)
        
        print(f"\nOriginal: '{original_transcription}'")
        print(f"Target: '{target_text}'")
        print(f"Starting attack (alpha={alpha}, iter={num_iterations}, SNR>={min_snr_db}dB, epsilon={epsilon.item():.6f})\n")
        
        best_adv_audio = None
        best_snr = -float('inf')
        attack_successful = False
        

        current_alpha = alpha
        best_loss = float('inf')
        plateau_counter = 0
        
        for iteration in range(num_iterations):
            # current_min_snr = min_snr_db - 5.0 * (1.0 - iteration / num_iterations)
            
            adv_audio.requires_grad = True
            loss = self.compute_loss(adv_audio, target_ids)
            
            self.model.zero_grad()
            if adv_audio.grad is not None:
                adv_audio.grad.zero_()
            
            loss.backward()
            grad = adv_audio.grad.data
            
            if loss.item() < best_loss - 0.01:
                best_loss = loss.item()
                plateau_counter = 0
            else:
                plateau_counter += 1
            

            if plateau_counter >= alpha_decay_patience:
                current_alpha *= alpha_decay_factor
                plateau_counter = 0
            
            adv_audio = adv_audio.detach() - current_alpha * grad.sign()
            perturbation = torch.clamp(adv_audio - original_audio, -epsilon, epsilon)
            adv_audio = original_audio + perturbation
            # adv_audio = self.project_perturbation(original_audio, adv_audio, current_min_snr)
            adv_audio = torch.clamp(adv_audio, -1.0, 1.0)
            
            check_now = loss.item() < 2.0 or (iteration + 1) % 10 == 0 or iteration == num_iterations - 1
            
            if check_now:
                with torch.no_grad():
                    current_snr = self.calculate_snr(original_audio, adv_audio)
                    adv_outputs = self.model.generate(adv_audio)
                    adv_transcription = self.processor.tokenizer.decode(adv_outputs[0], skip_special_tokens=True)
                    
                    is_match = adv_transcription.strip().lower() == target_text.strip().lower()
                    snr_ok = current_snr >= min_snr_db
                    
                    if (iteration + 1) % 100 == 0 or iteration == num_iterations - 1 or (is_match and snr_ok):
                        print(f"Iter {iteration + 1}: Loss={loss.item():.3f}, SNR={current_snr:.1f}dB, Alpha={current_alpha:.6f}, '{adv_transcription}'")
                    
                    if is_match and snr_ok:
                        attack_successful = True
                        best_snr = current_snr
                        best_adv_audio = adv_audio.clone()
                        
                        if early_stop:
                            print(f"Attack successful at iteration {iteration + 1}")
                            break
                    
                    if snr_ok and current_snr > best_snr:
                        best_snr = current_snr
                        best_adv_audio = adv_audio.clone()
        
        final_adv_audio = best_adv_audio if best_adv_audio is not None else adv_audio
        
        with torch.no_grad():
            final_outputs = self.model.generate(final_adv_audio)
            final_transcription = self.processor.tokenizer.decode(final_outputs[0], skip_special_tokens=True)
            final_snr = self.calculate_snr(original_audio, final_adv_audio)
        
        adversarial_audio_np = final_adv_audio.cpu().numpy().squeeze()
        
        attack_info = {
            'original_transcription': original_transcription,
            'target_transcription': target_text,
            'final_transcription': final_transcription,
            'snr_db': final_snr,
            'attack_successful': attack_successful
        }
        
        print(f"\nResults: SNR={final_snr:.2f}dB, Success={attack_successful}")
        print(f"Final: '{final_transcription}'")
        
        return adversarial_audio_np, attack_info
