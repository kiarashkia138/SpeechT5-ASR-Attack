import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import warnings

warnings.filterwarnings('ignore')


class CWAttackASR:
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
    
    def compute_loss(self, audio_values, target_ids):
        outputs = self.model(input_values=audio_values, labels=target_ids, return_dict=True)
        return outputs.loss
    
    def cw_loss(self, audio_values, original_audio, target_ids, c=1.0):
        attack_loss = self.compute_loss(audio_values, target_ids)
        perturbation_loss = torch.mean((audio_values - original_audio) ** 2)
        total_loss = c * attack_loss + perturbation_loss
        return total_loss, attack_loss, perturbation_loss
    
    def cw_attack(self, audio, target_text, sample_rate=16000, learning_rate=0.01,
                  num_iterations=1000, min_snr_db=20.0, binary_search_steps=5,
                  initial_const=1.0, abort_early=True, verbose=True):
        
        inputs = self.processor(audio=audio, sampling_rate=sample_rate, return_tensors="pt")
        audio_values = inputs.input_values.to(self.device)
        
        target_inputs = self.processor.tokenizer(target_text, return_tensors="pt")
        target_ids = target_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            original_outputs = self.model.generate(audio_values)
            original_transcription = self.processor.tokenizer.decode(original_outputs[0], skip_special_tokens=True)
        
        if verbose:
            print(f"\nOriginal: '{original_transcription}'")
            print(f"Target: '{target_text}'")
            print(f"Starting attack (lr={learning_rate}, iter={num_iterations}, SNR>={min_snr_db}dB)\n")
        
        lower_bound = 0.0
        upper_bound = 1e10
        const = initial_const
        
        best_adv_audio = None
        best_snr = -float('inf')
        best_const = const
        overall_best_adv = None
        overall_best_snr = -float('inf')
        attack_successful = False
        
        for search_step in range(binary_search_steps):
            if verbose:
                print(f"Search step {search_step + 1}/{binary_search_steps}, c={const:.4f}")
            
            w = torch.zeros_like(audio_values, requires_grad=True, device=self.device)
            optimizer = torch.optim.Adam([w], lr=learning_rate)
            search_successful = False
            
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                adv_audio = torch.tanh(w) + audio_values
                adv_audio = torch.clamp(adv_audio, -1.0, 1.0)
                
                total_loss, attack_loss, pert_loss = self.cw_loss(adv_audio, audio_values, target_ids, c=const)
                total_loss.backward()
                optimizer.step()
                
                if (iteration + 1) % 100 == 0 or iteration == num_iterations - 1:
                    with torch.no_grad():
                        test_adv_audio = torch.tanh(w) + audio_values
                        test_adv_audio = torch.clamp(test_adv_audio, -1.0, 1.0)
                        
                        current_snr = self.calculate_snr(audio_values, test_adv_audio)
                        adv_outputs = self.model.generate(test_adv_audio)
                        adv_transcription = self.processor.tokenizer.decode(adv_outputs[0], skip_special_tokens=True)
                        
                        is_match = adv_transcription.strip().lower() == target_text.strip().lower()
                        snr_ok = current_snr >= min_snr_db
                        
                        if verbose and ((iteration + 1) % 200 == 0 or iteration == num_iterations - 1 or (is_match and snr_ok)):
                            print(f"  Iter {iteration + 1}: Loss={total_loss.item():.3f}, SNR={current_snr:.1f}dB, '{adv_transcription}'")
                        
                        if is_match and snr_ok:
                            search_successful = True
                            if current_snr > best_snr:
                                best_snr = current_snr
                                best_adv_audio = test_adv_audio.clone().detach()
                                best_const = const
                        
                        if is_match and current_snr > overall_best_snr:
                            overall_best_snr = current_snr
                            overall_best_adv = test_adv_audio.clone().detach()
                            attack_successful = True
                        
                        if abort_early and is_match and snr_ok:
                            break
            
            if search_successful and best_snr >= min_snr_db:
                upper_bound = const
            else:
                lower_bound = const
            
            const = (lower_bound + upper_bound) / 2 if upper_bound < 1e10 else lower_bound * 10
        
        if best_adv_audio is not None:
            final_adv_audio = best_adv_audio
            final_snr = best_snr
        elif overall_best_adv is not None:
            final_adv_audio = overall_best_adv
            final_snr = overall_best_snr
        else:
            with torch.no_grad():
                final_adv_audio = torch.tanh(w) + audio_values
                final_adv_audio = torch.clamp(final_adv_audio, -1.0, 1.0)
            final_snr = self.calculate_snr(audio_values, final_adv_audio)
        
        adversarial_audio_np = final_adv_audio.detach().cpu().numpy().squeeze()
        
        with torch.no_grad():
            test_inputs = self.processor(audio=adversarial_audio_np, sampling_rate=sample_rate, return_tensors="pt")
            test_audio_values = test_inputs.input_values.to(self.device)
            
            final_outputs = self.model.generate(test_audio_values)
            final_transcription = self.processor.tokenizer.decode(final_outputs[0], skip_special_tokens=True)
            
            final_snr = self.calculate_snr(audio_values, test_audio_values)
        
        attack_info = {
            'original_transcription': original_transcription,
            'target_transcription': target_text,
            'final_transcription': final_transcription,
            'snr_db': final_snr,
            'attack_successful': attack_successful,
            'best_const': best_const
        }
        
        if verbose:
            print(f"\nResults: SNR={final_snr:.2f}dB, Success={attack_successful}")
            print(f"Final: '{final_transcription}'")
        
        return adversarial_audio_np, attack_info

