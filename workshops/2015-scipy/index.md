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

# Image Analysis in Python with SciPy and Scikit-Image

Presented by 
* **Stéfan van der Walt <stefanv@berkeley.edu>**
* **Joshua Warner <joshua.dale.warner@gmail.com>**
* **Steven Silvester <steven.silvester@ieee.org>**

<hr/>

This tutorial can be found online at [https://github.com/scikit-image/skimage-tutorials](https://github.com/scikit-image/skimage-tutorials)

Please launch the IPython notebook *from the root of the repository*.
<hr/>

From telescopes to satellite cameras to electron microscopes, scientists are producing more images than they can manually inspect. This tutorial will introduce automated image analysis using the "images as NumPy arrays" abstraction, run through various fundamental image analysis operations (filters, morphology, segmentation), and then focus on solving real-world problems through interactive demos.

Image analysis is central to a boggling number of scientific endeavors. Google needs it for their self-driving cars and to match satellite imagery and mapping data. Neuroscientists need it to understand the brain. NASA needs it to [map asteroids](http://www.bbc.co.uk/news/technology-26528516) and save the human race. It is, however, a relatively underdeveloped area of scientific computing.

The goal is for attendees to leave the tutorial confident of their ability to extract information from images in Python.

## Prerequisites

All of the below packages, including the non-Python ones, can be found in the [Anaconda](https://store.continuum.io/cshop/anaconda/) Python distribution, which can be obtained for free.
Alternatively, ``pip install [packagename]`` should work.

### Required packages

- scikit-image (0.11 or higher)

Required for scikit-image:

- Python (>=2.6 required, 3.4 recommended)
- numpy (>=1.6 required, 1.7 recommended)
- scipy (>=0.10 required, 0.13 recommended)

Required for image reading and viewing:

- matplotlib (>=1.1.0 required, 1.2 recommended)

### Example images

scikit-image ships with some example images in `skimage.data`. For this tutorial, we will additionally make use of images included in the `skimage-tutorials/images` folder. If you're reading this on your computer, you already have these images downloaded.

## Introduction

- [Gallery](http://scikit-image.org/docs/dev/auto_examples/), [User Guide](http://scikit-image.org/docs/stable/user_guide.html) and [API reference](http://scikit-image.org/docs/dev/api/api.html)
- Subpackages, see ``help(skimage)``

### The relationship of skimage with the Scientific Python eco-system

  - numpy (with skimage as the image processing layer)
  - scipy (in combination with ndimage)
  - sklearn (machine learning + feature extraction)
  - opencv for speed (e.g. in a factory setting)
  - novice (for teaching)

## Schedule

08:00--09:00 (Stéfan)

- [Images are just numpy arrays](../../lectures/00_images_are_arrays.ipynb)
- [Color and exposure](../../lectures/0_color_and_exposure.ipynb)

09:00--09:10 Break

09:15--10:15 (Steven)

- [Segmentation](../../lectures/4_segmentation.ipynb)

10:15--10:25 Snack break (snacks close at 10:30)

10:30--11:25 (Josh)

- [Stitching panoramas](../../lectures/adv3_panorama-stitching.ipynb)

11:30--12:00 (All)

Choose your own adventure!

- [Warping](../../lectures/adv4_warping.ipynb) -- **Mona Lisa de-warping competition inside!**
- [Analyzing micro arrays](../../lectures/adv2_microarray.ipynb)
- [Hands-on exercises](../../lectures/stackoverflow_challenges.ipynb) -- the best brain teasers from StackOverflow

---

Even more lectures [here](../../lectures) and [here](http://scipy-lectures.github.io).

---

+++

<div style="background: #DFD; border: 4px solid #EEE; border-radius: 6px; padding: 1em; font-size: 150%">
Please join us for the <a href="http://scipy2015.scipy.org/ehome/115969/259290/">scikit-image sprint</a> on Saturday and Sunday!
</div>

+++

## Further questions?

Feel free to grab hold of us during the conference, at the BoF session, or at the demo table!

Or meet the scikit-image team on <a href="https://gitter.im/scikit-image/scikit-image"><img src="https://badges.gitter.im/Join%20Chat.svg"/ style="display: inline"></a>

## Other ways to stay in touch

- Follow the project's progress [on GitHub](https://github.com/scikit-image/scikit-image).
- Ask the team questions on the [mailing list](https://groups.google.com/d/forum/scikit-image).
- [Contribute!](https://github.com/scikit-image/scikit-image/blob/main/CONTRIBUTING.txt)
- If you find it useful: please [cite our paper](https://peerj.com/articles/453/)!

---
