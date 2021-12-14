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

From telescopes to satellite cameras to electron microscopes, scientists are producing more images than they can manually inspect. This tutorial will introduce automated image analysis using the "images as numpy arrays" abstraction, run through various fundamental image analysis operations (filters, morphology, segmentation), and finally complete one or two more advanced real-world examples.

Image analysis is central to a boggling number of scientific endeavors. Google needs it for their self-driving cars and to match satellite imagery and mapping data. Neuroscientists need it to understand the brain. NASA needs it to [map asteroids](http://www.bbc.co.uk/news/technology-26528516) and save the human race. It is, however, a relatively underdeveloped area of scientific computing. Attendees will leave this tutorial confident of their ability to extract information from their images in Python.

# Prerequisites

All of the below packages, including the non-Python ones, can be found in the [Anaconda](https://store.continuum.io/cshop/anaconda/) Python distribution, which can (and should) be obtained for free. (Though some may need `conda install`ing.)

## Required packages

- scikit-image (0.10 or higher)

Required for scikit-image:

- Python (>=2.5 required, 2.7 recommended)
- numpy (>=1.6 required, 1.7 recommended)
- scipy (>=0.10 required, 0.13 recommended)

Required for image viewing and other examples:

- matplotlib (>=1.0 required, 1.2 recommended)

Required for skimage.viewer and skivi interactive examples

- Qt
- PyQt4/PySide

Required for development:

- cython (>=0.16 required, 0.19 recommended)

Recommended for IO:

- FreeImage
- Pillow/PIL

Recommended:

- PyAmg (Fast random-walker segmentation)

## Example images

scikit-image ships with some example images in `skimage.data`. For this tutorial, we will additionally make use of images included in the `images` directory of the `skimage-tutorials/scipy-2014` folder:

[Tutorial repository](https://github.com/scikit-image/skimage-tutorials)

# Sections

For convenience, we have divided this tutorial into several chapters, linked below. Throughout the tutorials, feel free to ask questions. We want you to come away confident in your image analysis skills!

- [Images are just NumPy arrays](../../lectures/00_images_are_arrays.ipynb)
- [Color and exposure](../../lectures/0_color_and_exposure.ipynb)
- [Image filters](../../lectures/1_image_filters.ipynb)
- [Feature detection](../../lectures/2_feature_detection.ipynb)
- [Morphological operations](../../lectures/3_morphological_operations.ipynb)
- [Segmentation](../../lectures/4_segmentation.ipynb)
- [Advanced example: measuring fluorescence intensity on chromosomes](../../lectures/adv0_chromosomes.ipynb)
- [Advanced example: measuring intensity along a microscopy image](../../lectures/adv1-lesion-quantification.ipynb)

# Improv time!

If we have time, raise your own analysis problems. They can make an interesting case study for the rest of the class! Even within the scikit-image team, we still surprise each other with the amazing diversity of applications we encounter!

# After the tutorial

Stay in touch!

- Follow the project's progress [on GitHub](https://github.com/scikit-image/scikit-image).
- Ask the team questions on the [mailing list](https://groups.google.com/d/forum/scikit-image)
- [Contribute!](https://github.com/scikit-image/scikit-image/blob/main/CONTRIBUTING.txt)
- If you find it useful: cite [our paper](https://peerj.com/articles/453/)!
