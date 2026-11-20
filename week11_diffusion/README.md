# Lecture
- **Slides:** https://disk.yandex.ru/d/k7onbN8RCWtZfg
- **Russian:** https://disk.yandex.ru/i/WWeUp70JNDjMvg
- **English:** https://cvpr2023-tutorial-diffusion-models.github.io , see [tutorial recording here](https://www.youtube.com/watch?v=1d4r19GEVos).



# Practice

Here's the basic [diffusion inference notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb) for reference.

This week's assignment is not about filling in a tutorial notebook but using a pre-existing code to create images with pre-trained diffusion models.

Choose a "protagonist": either an image of yourself / your friend / pet / a figurine or an obscure person/object of your choosing. The goal is to draw your 'protagonist' in imaginary surroundings.


We recommend you choose someone you personally know - not just "A warhammer space marine" or "Donald Trump". If you do choose a character you don't know personally, you have to prove (in a report) that the model doesn't already "know" this character without special training. For instance, don't choose Trump because most models can already generate Trump without special training.


The goal is to fine-tune Stable Diffusion (or a similar) open-source model to generate your likeness and edit your image using the tricks we covered in the lecture.


The main notebook for this assignment is DreamBooth - a technique for making diffusers learn specific:

- [DreamBooth starter notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb) (recommended)
- [optionally, instead of the previous one] train a larger model with LoRA [using this notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb).
- This is based on https://dreambooth.github.io . You may use other open-source implementations of DreamBooth. Proprietary models are fine for comparison, but at least some of the images you generated should use open-source models.

Gather a dataset with your own images, depending on the 'protagonist' you chose.

- You usually need 3-5 images for minimal working prototype, additional images sometimes help.
- The first notebook has instructions on how to upload them.
- Please avoid NSFW images and maintain basic decency. Humor and cringe is perfectly acceptable and mildly encouraged.

**Assignment 1 (3 points)** Train either version of dreambooth on your images and generate at least 3 different images with your protagonist. Multiple protagonists are fine, but not required.


**Assignment 2 (2 points)** Use inpainting to further modify details about the generated images. Create at least 3 changes (consecutive or to different images)

- See inpainting notebook https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/in_painting_with_stable_diffusion_using_diffusers.ipynb

- If you didn't complete the previous assignment, take real images of your protagonist and apply changes to them


You are expected to document the process: keep the intermediate images and prompts, write a short report of the summary of what you tried, what worked and what didn't. You don't have to write an essay, just make sure to demonstrate the intermediate steps.
