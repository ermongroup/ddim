# Denoising Diffusion Implicit Models (DDIM)

PyTorch implementation for the paper, by [Jiaming Song](http://tsong.me), Chenlin Meng and [Stefano Ermon](http://cs.stanford.edu/~ermon), 
Stanford Artificial Intelligence Laboratory

Implements sampling from an implicit model that is trained with the same procedure as [Denoising Diffusion Probabilistic Model](https://hojonathanho.github.io/diffusion/).


## Running the Experiments
The code has been tested on PyTorch 1.6.

### Train a model
Training is exactly the same as DDPM with the following:
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --ni
```

### Sampling from the model

#### Sampling from the generalized model for FID evaluation
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --fid --timesteps {STEPS} --eta {ETA} --ni
```
where 
- `ETA` controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM).
- `STEPS` controls how many timesteps used in the process.
- `MODEL_NAME` finds the pre-trained checkpoint according to its inferred path.

If you want to use the DDPM pretrained model:
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --use_pretrained --sample --fid --timesteps {STEPS} --eta {ETA} --ni
```
the `--use_pretrained` option will automatically load the model according to the dataset.

We provide a CelebA 64x64 model [here](https://drive.google.com/drive/folders/1si6Gn3U2ZrggnfYXOESWVFIwcNbyfh5o?usp=sharing), and use the DDPM version for CIFAR10 and LSUN.

If you want to use the version with the larger variance in DDPM: use the `--sample_type ddpm_noisy` option.

#### Sampling from the model for image inpainting 
Use `--interpolation` option instead of `--fid`.

#### Sampling from the sequence of images that lead to the sample
Use `--sequence` option instead.

The above two cases contain some hard-coded lines specific to producing the image, so modify them according to your needs.


## References and Acknowledgements

This implementation is based on
[https://github.com/hojonathanho/diffusion](diffusion) (the DDPM TensorFlow repo), 
[https://github.com/pesser/pytorch_diffusion](pytorch_diffusion) (PyTorch helper that loads the DDPM model), and
[https://github.com/ermongroup/ncsnv2](ncsnv2) (code structure).