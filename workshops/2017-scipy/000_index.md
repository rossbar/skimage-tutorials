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


# Image analysis in Python with SciPy and scikit-image

# Please clone
# https://github.com/scikit-image/skimage-tutorials

scikit-image is a collection of image processing algorithms for the
SciPy ecosystem.  It aims to have a Pythonic API (read: does what you'd expect), 
is well documented, and provides researchers and practitioners with well-tested,
fundamental building blocks for rapidly constructing sophisticated image
processing pipelines.

In this tutorial, we provide an interactive overview of the library,
where participants have the opportunity to try their hand at various
image processing challenges.

The tutorial consists of four parts, in which we:

1. give a general overview of the functionality available in the various submodules;
2. showcase analysis of real-world, anisotropic three-dimensional
   microscopy data, in celebration of the 0.13 release which greatly
   improves N-dimensional processing; 
3. demonstrate how to use scikit-image for machine learning tasks in
   combination with scikit-learn, and 
4. highlight interaction with other libraries, such as Keras and
   SciPy's LowLevelCallable.

Attendees are expected to have a working knowledge of NumPy, SciPy, and Matplotlib.

Across domains, modalities, and scales of exploration, images form an integral subset of scientific measurements. Despite a deep appeal to human intuition, gaining understanding of image content remains challenging, and often relies on heuristics (we hope to address some of that in part 3!). Even so, the wealth of knowledge contained inside of images cannot be understated, and <a href="http://scikit-image.org">scikit-image</a>, along with <a href="http://scipy.org">SciPy</a>, provides a strong foundation upon which to build algorithms and applications for exploring this domain.


# Prerequisites

All of the below packages, including the non-Python ones, can be found in the [Anaconda](https://store.continuum.io/cshop/anaconda/) Python distribution, which can be obtained for free. (Though some packages may need to be installed using `conda install packagename`.)

## Required packages

- scikit-image (0.13)

Required for scikit-image:

- Python >= 3.5
- numpy >= 1.13.1
- scipy >= 0.19

Required for image viewing and other examples:

- matplotlib >= 2

Required for machine learning section:

- scikit-learn >= 0.18

## Example images

scikit-image ships with some example images in `skimage.data`.

# Sections

For convenience, we have divided this tutorial into several chapters, linked below. Throughout the tutorials, feel free to ask questions. We want you to come away confident in your image analysis skills!

- 1:30 - 1:40: Introduction
- 1:40 - 2:10: [Images are just NumPy arrays](../../lectures/00_images_are_arrays.ipynb), [RANSAC](../lectures/5_ransac.ipynb)
- 2:15 - 3:30: [3D Image Processing](../../lectures/three_dimensional_image_processing.ipynb)
- 3:35 - 4:50: [Machine Learning](../../lectures/machine_learning.ipynb)
- 4:55 - 5:30: [Other Libraries](../../lectures/other_libraries.ipynb)

**Note:** Snacks are available 2:15-4:00; coffee & tea until 5.

# For later

- Check out the other [lectures](../../lectures)
- Try some [StackOverflow challenges](../../lectures/stackoverflow_challenges)

# Further information

- Here are some [real world use cases](http://bit.ly/skimage_real_world).
- [The other scikits](http://scikits.appspot.com) & interoperation with other Python packages
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
!pip install numpy --upgrade
```

```{code-cell} ipython3
import skimage
```

```{code-cell} ipython3
skimage.__version__
```
