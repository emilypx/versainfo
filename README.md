# Installing and running VERSA

To get our acoustic/audio quality metrics, I decided to use VERSA.

https://github.com/wavlab-speech/versa/tree/main

It's basically like an umbrella toolkit for running everyone else's audio quality metrics, which made it difficult to install and to get up and running. In this repo, I am including some files and this README.

## Part 1: Conda environment and installation

1. Create and activate conda environment with 3.10. Don't forget to load your conda or miniconda module first if that is how your computing cluster is set up.


```
conda create -n versa python=3.10 -y
conda activate versa
```

2. Install a CUDA-matched PyTorch FIRST, before installing versa itself. You can get your CUDA version with the `nvidia-smi` command. Here's how I installed it with CUDA 12.9 (see the 129 at the end of the URL), which is what we have on our cluster:

``pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu129``

3. Next install VERSA itself

```
git clone https://github.com/wavlab-speech/versa.git
cd versa
pip install .
```

4. Install some of the other models and tools for VERSA:

```
bash tools/setup_nisqa.sh
cd tools && bash install_srmr.sh && cd ..
``` 

5. At some point you need to get some other VERSA modules that don't get cloned with the normal cloning.

```
git submodule update --init --recursive
```

6. You can do this to confirm that you can actually import versa:

```
python -c "import versa; print('ok')"
python -c "from versa.bin.scorer import main; print('ok')"
```

You can look at the requirements I had when it worked for me in the `versa_pinned_requirements.txt` file, but everything should get installed properly, I think.

7. I had to set a lot of environment variables, and I needed a ton of help with that from Claude. You can find them all is `versa_env.sh`. You might not need to do this yourself, but I guess it can't hurt.

8. I had to apply some patches. It's possible you won't, if you get a lot of errors about soundfile backend and sox backend, you will need these patches. You'll see the two scripts in this repo. You can just run them as python scripts, and I think they will do the job.

```
apply_patches.py
apply_other_patch.py
```

9. Add you Hugging Face token to your profile on your cluster, which should make a lot of upcoming downloads for faster. I can never remember how to do this, so I always just look it up on the web.

## Part 2: Test run
Do a test run with a non-neural scorer and a neural scorer:

1. Make a sample scp file. You can use the `tiny.scp` file in the `test/temp` dir in this repo. It refers to a test audio file that comes with the versa distribution.
2. Make a sample yaml file for a non-neural scorer. You can use the `tiny.yaml` file in the `test/temp` dir in this repo. It refers to that `tiny.scp` file, above.
3. Make a sample yaml file for a neural scorer. You can use the `neural.yaml` file in the `test/temp` dir.
4. Make sure all the paths are correct in the yaml files and in `test.sh`.
5. Run the script `test.sh`, which you can also in the `test` dir.
6. I have provided my `.out` files so you can see the output I got. There are some warnings you can ignore.

## Part 3: Preliminary real run
1. Get a csv file that has the format `full-path-to-wav-file,transcript-of-that-wav-file` like we use when training wav2vec.
2. Create a new csv that is just the first 5 to 10 lines of that csv. This will just be to make sure it's working with real data, and it will give you a chance to say "y" whenever it asks you if you want to download something from somewhere.
3. Get these three files from this repo:

```
audio_quality_config.yaml
run_versa_quality.py
mini_run.sh
```
4. Edit `mini_run.sh` to make all the paths correct. You can see that I was trying it with some enenlhet data.
5. Make it executable and run it.
6. Hang out because you might have to say "y" to a lot of questions about downloading stuff.

## Part 4: Real run
Using `real_run.sh` as a guide, submit a full run as a job to your cluster with the full csv as an argument to `run_versa_quality.py`. Obviously you will need to update all your SBATCH lines to work with your cluster. It is important to give it plenty of CPUs. 



