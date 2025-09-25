from lightning_whisper_mlx import LightningWhisperMLX

whisper = LightningWhisperMLX(model="distil-large-v3", batch_size=12, quant=None)

text = whisper.transcribe(audio_path="/Users/jeffhaskin/Documents/Projects/Programs/lightning-owhisper-mlx/sample-sound.mp3")['text']

print(text)