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

Presented by **Stéfan van der Walt <stefan@sun.ac.za>**<br/>
Based on the SciPy2014 tutorial by **Juan Nunez-Iglesias** and **Tony S Yu**
<hr/>

This tutorial can be found online at [https://github.com/scikit-image/skimage-tutorials](https://github.com/scikit-image/skimage-tutorials)

<hr/>

From telescopes to satellite cameras to electron microscopes, scientists are producing more images than they can manually inspect. This tutorial will introduce automated image analysis using the "images as numpy arrays" abstraction, run through various fundamental image analysis operations (filters, morphology, segmentation), and finally complete one or two more advanced real-world examples.

Image analysis is central to a boggling number of scientific endeavors. Google needs it for their self-driving cars and to match satellite imagery and mapping data. Neuroscientists need it to understand the brain. NASA needs it to [map asteroids](http://www.bbc.co.uk/news/technology-26528516) and save the human race. It is, however, a relatively underdeveloped area of scientific computing. Attendees will leave this tutorial confident of their ability to extract information from their images in Python.

## Prerequisites

All of the below packages, including the non-Python ones, can be found in the [Anaconda](https://store.continuum.io/cshop/anaconda/) Python distribution, which can be obtained for free.

### Required packages

- scikit-image (0.10 or higher)

Required for scikit-image:

- Python (>=2.5 required, 2.7 recommended)
- numpy (>=1.6 required, 1.7 recommended)
- scipy (>=0.10 required, 0.13 recommended)

Required for image reading and viewing:

- matplotlib (>=1.0 required, 1.2 recommended)

### Example images

scikit-image ships with some example images in `skimage.data`. For this tutorial, we will additionally make use of images included in the `images` directory of the `skimage-tutorials/images` folder.

## Introduction

- [Gallery](http://scikit-image.org/docs/dev/auto_examples/) -- soon to be interactive ([preview here](http://sharky93.github.io/docs/gallery/auto_examples))!
- Subpackages, see ``skimage.<TAB>``

## Sections

For convenience, we have divided this tutorial into several chapters, linked below. Throughout the tutorials, feel free to ask questions. We want you to come away confident in your image analysis skills!

- [Images are just numpy arrays](../../lectures/00_images_are_arrays.ipynb)
- [Color and exposure](../../lectures/0_color_and_exposure.ipynb) [briefly cover]
- [Segmentation](../../lectures/4_segmentation.ipynb)
- [Real world example: panorama stitching](../../lectures/example_pano.ipynb)
- Exercises and further demos, if there's time

More lectures [here](../../lectures) and [here](http://scipy-lectures.github.io).

## Improv time!

If we have time, raise your own analysis problems. They can make an interesting case study for the rest of the class! Even within the scikit-image team, we still surprise each other with the amazing diversity of applications we encounter.

## After the tutorial

Stay in touch!

- Follow the project's progress [on GitHub](https://github.com/scikit-image/scikit-image).
- Ask the team questions on the [mailing list](https://groups.google.com/d/forum/scikit-image)
- [Contribute!](https://github.com/scikit-image/scikit-image/blob/main/CONTRIBUTING.txt)
- If you find it useful: cite [our paper](https://peerj.com/articles/453/)!
