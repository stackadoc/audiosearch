# audiosearch, by Stackadoc
Silence ! Let's the (binary) music begin

## Technical music sheet

This process is divided into the following sections:

- Environment setup.
- Audio Data-Preprocessing & Populating the Qdrant Vector Database.
- Gradio Interface setup.
- Testing Text to audio Search

## Install Docker (if you haven’t already).

Please use the great documentation of our friends at digital Ocean !

**Here’s the Ubuntu 22.04 (but it surely exists for your own flavour)**

https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-22-04

## Orchestrating Qdrant

Kicking off the setup for this musical project starts with retrieving the Docker container image imbued with melodious elements, followed by deploying it on your local Docker daemon. (Make sure to orchestrate the Docker application before proceeding.)

Harmonize your system by pulling the Qdrant client container, a symphony of data, from the Docker Hub repository. Then, conduct the container with the following command, setting the stage for the application to perform at **`localhost:6333`**, a digital concert hall for your project's operatic debut.

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
-v $(pwd)/qdrant_storage:/qdrant/storage:z \
qdrant/qdrant 
```

*NOTE: If you are running on Windows, kindly replace $(pwd) with your local path.*

# The Python code

Setup your environment, install the requirements and 

```bash
git clone git@github.com:stackadoc/musicsearch.git
cd musicsearch
virtualenv --python 3.10 venv
source venv/bin/activate
pip install -r requirements.txt
```

# **Data Pre-Processing and Populating the Vector Database**

## Download the demo samples

For this project I’ve used the https://www.kaggle.com/competitions/park-spring-2023-music-genre-recognition/data dataset, which is a collection of 4500 files split into 10 music genres categories

1. First let’s install the Kaggle package; for that open Jupyter Notebook in vscode and install the package using pip install kaggle
2. Obtain your Kaggle API key: You can generate it on Kaggle by going to your account settings and under the ‘API’ section, click on ‘Create New API Token’. This will download a file named ‘kaggle.json’ which holds the credentials required.
3. Move the downloaded ‘kaggle.json’ file to your project directory.
4. Open the terminal and run the following command to download the dataset above mentioned: `kaggle competitions download -c park-spring-2023-music-genre-recognition`
5. After downloading, you may need to unzip or extract the contents of the downloaded file for further processing
6. Copy and store the folder of training, where every genre subfolders are displayed :

```bash
# For me, it gives the following :
KAGGLE_DB_PATH = '/home/arthur/data/kaggle/park-spring-2023-music-genre-recognition/train/train'
```

## Populate the demo database, using database.py

Simply modify the parameters for your CACHE_FOLDER and KAGGLE_DB_PATH, in the [database.py](http://database.py) folder

## Simply launch the Gradio app and enjoy !

Here’s the simple code : 

```bash
python app.py
```

Go on your favorite webbrowser and open the local URL displayed in your execution stack (for me, Gradio opens at : http://127.0.0.1:7861/ )