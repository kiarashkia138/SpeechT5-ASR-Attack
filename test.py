from transformers import pipeline,AutoProcessor, AutoModelForSpeechSeq2Seq

pipe = pipeline("automatic-speech-recognition", model="microsoft/speecht5_asr")
processor = AutoProcessor.from_pretrained("microsoft/speecht5_asr")
model = AutoModelForSpeechSeq2Seq.from_pretrained("microsoft/speecht5_asr")

def gen_text(audio_path):
    result = pipe(audio_path)
    print(result["text"])

if __name__ == "__main__":
    gen_text("adversarial_audio_cw.wav")
    gen_text("adversarial_audio_pgd.wav")