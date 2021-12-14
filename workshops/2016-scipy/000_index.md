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

Across domains, modalities, and scales of exploration, images form an integral subset of scientific measurements. Despite a deep appeal to human intuition, gaining understanding of image content remains challenging, and often relies on heuristics. Even so, the wealth of knowledge contained inside of images cannot be understated. <a href="http://scikit-image.org">Scikit-image</a> is an image processing library, built on top of <a href="http://scipy.org">SciPy</a>, that provides researchers, practitioners, and educators access to a strong foundation upon which to build algorithms and applications.

Image analysis is central to a boggling number of scientific endeavors. Google needs it for their self-driving cars and to match satellite imagery and mapping data. Neuroscientists need it to understand the brain. NASA needs it to [map asteroids](http://www.bbc.co.uk/news/technology-26528516) and save the human race. It is, however, a relatively underdeveloped area of scientific computing. Attendees will leave this tutorial confident of their ability to extract information from their images in Python.

The tutorial is aimed at intermediate users of scientific Python with a *working knowledge of NumPy*.  We introduce the library and give practical, real-world examples of applications. Throughout, attendees are given the opportunity to learn through hands-on exercises.


# Prerequisites

All of the below packages, including the non-Python ones, can be found in the [Anaconda](https://store.continuum.io/cshop/anaconda/) Python distribution, which can be obtained for free. (Though some packages may need to be installed using `conda install packagename`.)

## Required packages

- scikit-image (0.12.3)

Required for scikit-image:

- Python >= 3.5
- numpy >= 1.10
- scipy >= 0.17

Required for image viewing and other examples:

- matplotlib >= 1.5

Required for development (not this tutorial):

- cython >= 0.23

Recommended for Input/Output:

- Pillow >= 3.0

## Example images

scikit-image ships with some example images in `skimage.data`.

# Sections

For convenience, we have divided this tutorial into several chapters, linked below. Throughout the tutorials, feel free to ask questions. We want you to come away confident in your image analysis skills!

- 1:30 - 1:40: Introduction
- 1:40 - 2:20: [Images are just NumPy arrays](../../lectures/00_images_are_arrays.ipynb)
- [10 minute break]
- 2:30 - 2:50: [Image filters](../../lectures/1_image_filters.ipynb)
- 2:50 - 3:30: [Segmentation](../../lectures/4_segmentation.ipynb)
- [20 minute break]
- 3:50 - 4:30: [RANSAC](../../lectures/5_ransac.ipynb)
- [10 minute break]
- 4:40 - 5:30: [Geometric transforms and warping](../../lectures/adv4_warping.ipynb), [Panorama stitching](../../lectures/solutions/adv3_panorama-stitching-solution.ipynb)

# Further information

- Here are some [real world use cases](http://bit.ly/skimage_real_world).
- [The other scikits](http://scikits.appspot.com) & interoperation with other Python packages
- ndimage vs scikit-image
- scikit-image vs opencv


# After the tutorial

Stay in touch!

- Follow the project's progress [on GitHub](https://github.com/scikit-image/scikit-image).
- Ask the team questions on the [mailing list](https://groups.google.com/d/forum/scikit-image)
- [Contribute!](https://github.com/scikit-image/scikit-image/blob/main/.github/CONTRIBUTING.txt)
- Read [our paper](https://peerj.com/articles/453/)!
