## Installing and running VERSA

To get our acoustic/audio quality metrics, I decided to use VERSA.

https://github.com/wavlab-speech/versa/tree/main

It's basically like an umbrella toolkit for running everyone else's audio quality metrics, which made it difficult to install and to get up and running. In this repo, I am including some files and this README.

## Step 1: Environment and installation

1. Create and activate conda environment with 3.10. Don't forget to load your conda or miniconda module first if that is how your compuyting cluster is set up.

``conda create -n versa python=3.10 -y``
``conda activate versa``

2. Install a CUDA-matched PyTorch FIRST, before installing versa itself. You can get your CUDA version with the `nvidia-smi` command. Here's how I installed it with CUDA 12.9:

``pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu129``

3. Then install VERSA itself

```
git clone https://github.com/wavlab-speech/versa.git
cd versa
pip install .
```

4. Install some of the other models and tools

```
bash tools/setup_nisqa.sh
cd tools && bash install_srmr.sh && cd ..
``` 

5. At some point you need to get some other modules that don't get cloned with the normal cloning.

```
git submodule update --init --recursive
```
