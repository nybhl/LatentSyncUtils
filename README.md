HOW TO INSTALL LATENTSYNC
if you have 8gb vram then just clone this repo , cd Latentsync


clone the latentsync repository in hugginface(1.6) and rename that as checkpoints


create a venv/conda environment and pip install -r requirements.txt (comment out torch, torchvision to your need). I installed myself similar to other projects to make sure cudnn works

there will be some dependency conflicts like mediapipe,xformers just install the latest versions 

you are good to go you can run gradio_app.py or inference.sh
I use ./batch_inference.sh --random_seed=0
