from copy import deepcopy

import librosa
import numpy as np
import sounddevice as sd
# Load the CLAP model and processor
from transformers import pipeline

audio_classifier = pipeline(task="zero-shot-audio-classification",
                            model="laion/larger_clap_general")

soundscape_taxonomy = {
    'clap': {}, "music": {}, "silence": {}, "people talking": {},
}

# Function to capture audio
def capture_audio(duration=1):Coucouc
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32', device=5)
    sd.wait()  # Wait until recording is finished
    return np.squeeze(audio)


# Function to classify audio
def classify_audio_rec(audio, fs=48000):
    resampled_audio = librosa.resample(audio, orig_sr=16000, target_sr=fs)
    st = deepcopy(soundscape_taxonomy)

    def recursive_classif(resampled_audio, st, candidates):
        results = audio_classifier(resampled_audio, candidate_labels=candidates)
        if type(st[results[0]['label']]) == dict:
            return recursive_classif(
                resampled_audio, st=st[results[0]['label']], candidates=st[results[0]['label']].keys()
            )
        elif type(st[results[0]['label']]) is list:
            return audio_classifier(resampled_audio, candidate_labels=candidates)

    return recursive_classif(resampled_audio, st, candidates=st.keys())

def classify_audio(audio):
    if type(audio[0]) == np.int16:
        audio = audio.astype(float)
        audio /= np.max(np.abs(audio))
    resampled_audio = librosa.resample(audio, orig_sr=16000, target_sr=48000)
    label = audio_classifier(resampled_audio, candidate_labels=soundscape_taxonomy.keys())
    return label

if __name__ == '__main__':
    # Main loop to capture and classify audio every 5 seconds
    print("Listening...")
    try:
        while True:
            audio = capture_audio()
            resampled_audio = librosa.resample(audio, orig_sr=16000, target_sr=48000)
            label = audio_classifier(resampled_audio, candidate_labels=soundscape_taxonomy.keys())
            print(f"Detected sound: {label[0]['label']} - {label[0]['score']}")
    except KeyboardInterrupt:
        print("Stopped by user")
