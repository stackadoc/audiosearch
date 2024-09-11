import hashlib
import time

import gradio as gr
import numpy as np
import random

from reverse_audio_search.sound_event_detection import classify_audio


def add_to_stream(audio, instream):
    time.sleep(1)
    if audio is None:
        return gr.update(), instream
    if instream is None:
        ret = audio
    else:
        ret = (audio[0], np.concatenate((instream[1], audio[1])))
    return classify_audio(ret[1]), ret

with gr.Blocks() as demo:
    inp = gr.Audio(sources=["microphone"])
    out = gr.Text("Sound Event Detection")
    stream = gr.State()
    clear = gr.Button("Clear")

    inp.stream(add_to_stream, [inp, stream], [out, stream])
    clear.click(lambda: [None, None, None], None, [inp, out, stream])

if __name__ == "__main__":
    demo.launch()
