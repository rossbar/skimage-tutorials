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

# Parsing pixels with scikit-image

Presented by **Stéfan van der Walt <stefanv@berkeley.edu>**<br/>
<hr/>

This tutorial can be found online at [https://github.com/scikit-image/skimage-tutorials](https://github.com/scikit-image/skimage-tutorials)

Please launch the IPython notebook *from the root of the repository*.
<hr/>

From telescopes to satellite cameras to electron microscopes, scientists are producing more images than they can manually inspect. This tutorial will introduce automated image analysis using the "images as numpy arrays" abstraction, run through various fundamental image analysis operations (filters, morphology, segmentation), and finally give you a chance to try your hand at some real-world examples from StackOverflow.

Image analysis is central to a boggling number of scientific endeavors. Google needs it for their self-driving cars and to match satellite imagery and mapping data. Neuroscientists need it to understand the brain. NASA needs it to [map asteroids](http://www.bbc.co.uk/news/technology-26528516) and save the human race. It is, however, a relatively underdeveloped area of scientific computing.

The goal is for attendees to leave the tutorial confident of their ability to extract information from images in Python.

## Prerequisites

All of the below packages, including the non-Python ones, can be found in the [Anaconda](https://store.continuum.io/cshop/anaconda/) Python distribution, which can be obtained for free.
Alternatively, under OSX, a ``pip install`` should work.

### Required packages

- scikit-image (0.10 or higher)

Required for scikit-image:

- Python (>=2.5 required, 2.7 recommended)
- numpy (>=1.6 required, 1.7 recommended)
- scipy (>=0.10 required, 0.13 recommended)

Required for image reading and viewing:

- matplotlib (>=1.0 required, 1.2 recommended)

### Example images

scikit-image ships with some example images in `skimage.data`. For this tutorial, we will additionally make use of images included in the `skimage-tutorials/images` folder.

## Introduction

- [Gallery](http://scikit-image.org/docs/dev/auto_examples/) -- soon to be interactive ([preview here](http://sharky93.github.io/docs/gallery/auto_examples))!
- Subpackages, see ``help(skimage)``

## Sections

For convenience, we have divided this tutorial into several chapters, linked below.

I am going to move through the material fairly quickly so that you can start playing around with code as soon as possible.  **But that means you have to stop me if something is not clear!**  My goal is for you to come away confident that you can manipulate images in Python.

- [Images are just numpy arrays](../../lectures/00_images_are_arrays.ipynb)
- [Color and exposure](../../lectures/0_color_and_exposure.ipynb)
- [Segmentation](../../lectures/4_segmentation.ipynb) [depending on whether we have time]
- [Real world example: analyzing microarrays](../../lectures/adv2_microarray.ipynb)
- [Hands-on exercizes](../../lectures/stackoverflow_challenges.ipynb), questions and discussion!

More lectures [here](../../lectures) and [here](http://scipy-lectures.github.io).

## Further questions?

Feel free to grab hold of me tomorrow during the conference, or at the [PyData Ask Us Anything](http://strataconf.com/big-data-conference-ca-2015/public/schedule/detail/41087)!

Or meet the scikit-image team on <a href="https://gitter.im/scikit-image/scikit-image?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge"><img src="https://badges.gitter.im/Join%20Chat.svg"/ style="display: inline"></a>

## Other ways to stay in touch

- Follow the project's progress [on GitHub](https://github.com/scikit-image/scikit-image).
- Ask the team questions on the [mailing list](https://groups.google.com/d/forum/scikit-image)
- [Contribute!](https://github.com/scikit-image/scikit-image/blob/main/CONTRIBUTING.txt)
- If you find it useful: cite [our paper](https://peerj.com/articles/453/)!

+++

---

## The relationship of skimage with the Scientific Python eco-system

  - numpy (with skimage as the image processing layer)
  - scipy (in combination with ndimage)
  - sklearn (machine learning + feature extraction)
  - opencv for speed (e.g. in a factory setting)
  - novice (for teaching)

---

<div style="height: 400px;"></div>

```{code-cell} ipython3
%reload_ext load_style
%load_style ../themes/tutorial.css
```
