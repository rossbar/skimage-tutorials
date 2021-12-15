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

<div style="border: solid 1px; background: #abcfef; font-size: 150%; padding: 1em; margin: 1em; width: 75%;">

<p>To participate, please follow the preparation instructions at</p>
<p>https://github.com/scikit-image/skimage-tutorials/</p>
<p>(click on **preparation.md**).</p>

</div>

<hr/>
TL;DR: Install Python 3.6, scikit-image, and the Jupyter notebook.  Then clone this repo:

```python
git clone --depth=1 https://github.com/scikit-image/skimage-tutorials
```
<hr/>

scikit-image is a collection of image processing algorithms for the
SciPy ecosystem.  It aims to have a Pythonic API (read: does what you'd expect), 
is well documented, and provides researchers and practitioners with well-tested,
fundamental building blocks for rapidly constructing sophisticated image
processing pipelines.

In this tutorial, we provide an interactive overview of the library,
where participants have the opportunity to try their hand at various
image processing challenges.

Attendees are expected to have a working knowledge of NumPy, SciPy, and Matplotlib.

Across domains, modalities, and scales of exploration, images form an integral subset of scientific measurements. Despite a deep appeal to human intuition, gaining understanding of image content remains challenging, and often relies on heuristics. Even so, the wealth of knowledge contained inside of images cannot be understated, and <a href="http://scikit-image.org">scikit-image</a>, along with <a href="http://scipy.org">SciPy</a>, provides a strong foundation upon which to build algorithms and applications for exploring this domain.


# Prerequisites

Please see the [preparation instructions](https://github.com/scikit-image/skimage-tutorials/blob/main/preparation.md).

# Schedule

- 1:00–1:50: Introduction & [images as NumPy arrays](../../lectures/00_images_are_arrays.ipynb)
- 2:00–2:50: [Filters](../../lectures/1_image_filters.ipynb)
- 3:10–3:30: [Morphology](../../lectures/3_morphological_operations.ipynb)
- 3:30–4:00: [Segmentation](../../lectures/4_segmentation.ipynb)
- 4:00–4:20: [StackOverflow Challenges | BYO problem](../../lectures/stackoverflow_challenges.ipynb) / Q&A

# For later

- Check out the other [lectures](../../lectures)
- Check out a [3D segmentation workflow](../../lectures/three_dimensional_image_processing.ipynb)
- Some [real world use cases](http://bit.ly/skimage_real_world)


# After the tutorial

Stay in touch!

- Follow the project's progress [on GitHub](https://github.com/scikit-image/scikit-image).
- Ask questions on [image.sc](https://forum.image.sc) (don't forget to tag with #skimage or #scikit-image!)
- Ask the team questions on the [mailing list](https://mail.python.org/mailman/listinfo/scikit-image)
- [Contribute!](http://scikit-image.org/docs/dev/contribute.html)
- Read (and cite!) [our paper](https://peerj.com/articles/453/) (or [this other paper, for skimage in microtomography](https://ascimaging.springeropen.com/articles/10.1186/s40679-016-0031-0))

```{code-cell} ipython3
%run ../../check_setup.py
```