import gradio as gr
from qdrant_client import QdrantClient
from transformers import ClapModel, ClapProcessor

# Loading the Qdrant DB in local ###################################################################
client = QdrantClient("localhost", port=6333)
print("[INFO] Client created...")

# loading the model
print("[INFO] Loading the model...")
model_name = "laion/larger_clap_general"
model = ClapModel.from_pretrained(model_name)
processor = ClapProcessor.from_pretrained(model_name)

# Gradio Interface #################################################################################
max_results = 10


def sound_search(query):
    text_inputs = processor(text=query, return_tensors="pt")
    text_embed = model.get_text_features(**text_inputs)[0]

    hits = client.search(
        collection_name="demo_db",
        query_vector=text_embed,
        limit=max_results,
    )
    return [
        gr.Audio(
            hit.payload['audio_path'],
            label=f"style: {hit.payload['style']} -- score: {hit.score}")
        for hit in hits
    ]


with gr.Blocks() as demo:
    gr.Markdown(
        """# Sound search database """
    )
    inp = gr.Textbox(placeholder="What sound are you looking for ?")
    out = [gr.Audio(label=f"{x}") for x in range(max_results)]  # Necessary to have different objs
    inp.change(sound_search, inp, out)

demo.launch()
