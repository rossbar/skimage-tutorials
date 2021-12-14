---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# scikit-image: Image Processing in Python

# https://github.com/scikit-image/skimage-tutorials


0. If you haven't yet, please clone this repository:

```
git clone --depth=1 https://github.com/scikit-image/skimage-tutorials
```

On conference wifi, this will take about 5 minutes :/  You can also copy a cloned copy from a friend via, e.g., a USB stick.


1. If you already have, please pull the latest updates:

```
git pull
```

## Abstract

Across domains, modalities, and scales of exploration, images form an integral subset of scientific measurements. Despite a deep appeal to human intuition, gaining understanding of image content remains challenging; yet the wealth of knowledge contained inside of images cannot be overstated.

`scikit-image` is a collection of image processing algorithms for the Scientific Python (SciPy) ecosystem. It has a Pythonic API, is well documented, and aims to provide researchers and practitioners with well-tested, fundamental building blocks for rapidly constructing sophisticated image processing pipelines.

In this training, we give an interactive overview of the library, and let participants try their hand at various educational image processing challenges.

Attendees are expected to have a working knowledge of NumPy, SciPy, and Matplotlib.

## Presenter Bio

Stéfan van der Walt is a researcher at the Berkeley Institute for Data Science, UC Berkeley. He has been a part of the scientific Python developer community since 2006, and is the founder of scikit-image. Outside of work, he enjoys running, music, the great outdoors, and being a dad.


# Prerequisites

Please refer to the [preparation instructions](https://github.com/scikit-image/skimage-tutorials/blob/main/preparation.md).


# Sections

Throughout the tutorial, please feel free to interrupt with questions.  The roadmap below is a guideline only.

- 2:00 - 2:30:
  - Introduction
  - [Images are just NumPy arrays](../../lectures/00_images_are_arrays.ipynb) + exercises
- 3:00 - 3:30: [SciPy Quiz: Know your toolbox!](../../lectures/numpy_scipy_quiz.ipynb)
- 3:30 - 4:15:
   - [Filters](../../lectures/1_image_filters.ipynb) + exercises
- 4:15 - 5:15:
   - [3D segmentation](../../lectures/three_dimensional_image_processing.ipynb) + exercises
   - [RANSAC](../../lectures/5_ransac.ipynb)
- 5:15 - 5:45: [Interlude: Neural Networks, Parallelization, SciPy LowLevelCallable](../../lectures/other_libraries.ipynb)
- 5:45 - 6:00: Q&A, General Discussion

Optional extra: [stitching image panoramas](../../lectures/adv3_panorama-stitching.ipynb)


# For later

- Check out the other [scikit-image lectures](../../lectures)
- Try some [StackOverflow challenges](../../lectures/stackoverflow_challenges)

# Further information

- Here are some [real world use cases](http://bit.ly/skimage_real_world).
- ndimage vs scikit-image
- scikit-image vs opencv


# After the tutorial

Stay in touch!

- Follow the project's progress [on GitHub](https://github.com/scikit-image/scikit-image).
- Ask the team questions on the [mailing list](https://mail.python.org/mailman/listinfo/scikit-image)
- [Contribute!](https://github.com/scikit-image/scikit-image/blob/main/.github/CONTRIBUTING.txt)
- Read [our paper](https://peerj.com/articles/453/) (or [this one, for skimage in microscopy](https://ascimaging.springeropen.com/articles/10.1186/s40679-016-0031-0))

```{code-cell} ipython3
import numpy as np
np.__version__
```

```{code-cell} ipython3
import skimage
```

```{code-cell} ipython3
skimage.__version__
```
