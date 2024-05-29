import gc
import hashlib
import os
from glob import glob

import librosa
import torch
from diskcache import Cache
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

# PARAMETERS #######################################################################################
CACHE_FOLDER = '/home/arthur/data/music/demo_audio_search/audio_embeddings_cache_individual/'
KAGGLE_DB_PATH = '/home/arthur/data/kaggle/park-spring-2023-music-genre-recognition/train/train'


# Functions utils ##################################################################################
def get_md5(fpath):
    with open(fpath, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def get_audio_embedding(model, audio_file, cache):
    # Compute a unique hash for the audio file
    file_key = f"{model.config._name_or_path}" + get_md5(audio_file)
    if file_key in cache:
        # If the embedding for this file is cached, retrieve it
        embedding = cache[file_key]
    else:
        # Otherwise, compute the embedding and cache it
        y, sr = librosa.load(audio_file, sr=48000)
        inputs = processor(audios=y, sampling_rate=sr, return_tensors="pt")
        embedding = model.get_audio_features(**inputs)[0]
        gc.collect()
        torch.cuda.empty_cache()
        cache[file_key] = embedding
    return embedding



# ################## Loading the CLAP model ###################
# loading the model
print("[INFO] Loading the model...")
model_name = "laion/larger_clap_general"
model = ClapModel.from_pretrained(model_name)
processor = ClapProcessor.from_pretrained(model_name)

# Initialize the cache
os.makedirs(CACHE_FOLDER, exist_ok=True)
cache = Cache(CACHE_FOLDER)

# Creating a qdrant collection #####################################################################
client = QdrantClient("localhost", port=6333)
print("[INFO] Client created...")

print("[INFO] Creating qdrant data collection...")
if not client.collection_exists("demo_db"):
    client.create_collection(
        collection_name="demo_db",
        vectors_config=models.VectorParams(
            size=model.config.projection_dim,
            distance=models.Distance.COSINE
        ),
    )

# Embed the audio files !
audio_files = [p for p in glob(os.path.join(KAGGLE_DB_PATH, '*/*.wav'))]
chunk_size, idx = 1, 0
total_chunks = int(len(audio_files) / chunk_size)

# Use tqdm for a progress bar
for i in tqdm(range(0, len(audio_files), chunk_size), total=total_chunks):
    chunk = audio_files[i:i + chunk_size]  # Get a chunk of audio files
    records = []
    print("[INFO] Uploading data records to data collection...")
    for audio_file in chunk:
        embedding = get_audio_embedding(model, audio_file, cache)
        records.append(
            models.PointStruct(
                id=idx, vector=embedding,
                payload={
                    "audio_path": audio_file,
                    "style": audio_file.split('/')[-1]}
            )
        )
        idx += 1
    print(f'Uploading batch starting at idx {i}')
    client.upload_points(
        collection_name="synthia_db",
        points=records
    )
print("[INFO] Successfully uploaded data records to data collection!")


# It's a good practice to close the cache when done
cache.close()