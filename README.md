# Denoising Diffusion Implicit Models (DDIM)

[Jiaming Song](http://tsong.me), [Chenlin Meng](http://cs.stanford.edu/~chenlin) and [Stefano Ermon](http://cs.stanford.edu/~ermon), Stanford

Implements sampling from an implicit model that is trained with the same procedure as [Denoising Diffusion Probabilistic Model](https://hojonathanho.github.io/diffusion/), but costs much less time and compute if you want to sample from it (click image below for a video demo):

<a href="http://www.youtube.com/watch?v=WCKzxoSduJQ" target="_blank">![](http://img.youtube.com/vi/WCKzxoSduJQ/0.jpg)</a>

## **Integration with 🤗 Diffusers library**

DDIM is now also available in 🧨 Diffusers and accesible via the [DDIMPipeline](https://huggingface.co/docs/diffusers/api/pipelines/ddim).
Diffusers allows you to test DDIM in PyTorch in just a couple lines of code.

You can install diffusers as follows:

```
pip install diffusers torch accelerate
```

And then try out the model with just a couple lines of code:

```python
from diffusers import DDIMPipeline

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
image = ddim(num_inference_steps=50).images[0]

# save image
image.save("ddim_generated_image.png")
```

More DDPM/DDIM models compatible with hte DDIM pipeline can be found directly [on the Hub](https://huggingface.co/models?library=diffusers&sort=downloads&search=ddpm)

To better understand the DDIM scheduler, you can check out [this introductionary google colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)

The DDIM scheduler can also be used with more powerful diffusion models such as [Stable Diffusion](https://huggingface.co/docs/diffusers/v0.7.0/en/api/pipelines/stable_diffusion#stable-diffusion-pipelines)

You simply need to [accept the license on the Hub](https://huggingface.co/runwayml/stable-diffusion-v1-5), login with `huggingface-cli login` and install transformers:

```
pip install transformers
```

Then you can run:

```python
from diffusers import StableDiffusionPipeline, DDIMScheduler

ddim = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=ddim)

image = pipeline("An astronaut riding a horse.").images[0]

image.save("astronaut_riding_a_horse.png")
```

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

We provide a CelebA 64x64 model [here](https://drive.google.com/file/d/1R_H-fJYXSH79wfSKs9D-fuKQVan5L-GR/view?usp=sharing), and use the DDPM version for CIFAR10 and LSUN.

If you want to use the version with the larger variance in DDPM: use the `--sample_type ddpm_noisy` option.

#### Sampling from the model for image inpainting 
Use `--interpolation` option instead of `--fid`.

#### Sampling from the sequence of images that lead to the sample
Use `--sequence` option instead.

The above two cases contain some hard-coded lines specific to producing the image, so modify them according to your needs.


## References and Acknowledgements
```
@article{song2020denoising,
  title={Denoising Diffusion Implicit Models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv:2010.02502},
  year={2020},
  month={October},
  abbr={Preprint},
  url={https://arxiv.org/abs/2010.02502}
}
```


This implementation is based on / inspired by:

- [https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) (the DDPM TensorFlow repo), 
- [https://github.com/pesser/pytorch_diffusion](https://github.com/pesser/pytorch_diffusion) (PyTorch helper that loads the DDPM model), and
- [https://github.com/ermongroup/ncsnv2](https://github.com/ermongroup/ncsnv2) (code structure).
