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

If you cloned it before today, use `git pull origin` to get the latest changes.

```{code-cell} ipython3
%run ../../check_setup.py
```

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

- 1:30–2:20: Introduction & [images as NumPy arrays](../../lectures/00_images_are_arrays.ipynb)
- 2:30–3:20: [Filters](../../lectures/1_image_filters.ipynb)
- 3:30–4:20: [Segmentation](../../lectures/4_segmentation.ipynb)
- 4:30–5:00: [Advanced workflow example](../../lectures/adv5-pores.ipynb)
- 5:00–5:20: [Tour of scikit-image](../../lectures/tour_of_skimage.ipynb)
- 5:20–5:30: Q&A

**Note:** Snacks are available 2:15-4:00; coffee & tea until 5.

# For later

- Check out the other [lectures](../../lectures)
- Check out a [3D segmentation workflow](../../lectures/three_dimensional_image_processing.ipynb)
- Some [real world use cases](http://bit.ly/skimage_real_world)


# After the tutorial

Stay in touch!

- Come to the [sprint](https://www.scipy2019.scipy.org/sprints-schedule)!
- Follow the project's progress [on GitHub](https://github.com/scikit-image/scikit-image).
- Ask the team questions on the [mailing list](https://mail.python.org/mailman/listinfo/scikit-image)
- [Contribute!](http://scikit-image.org/docs/dev/contribute.html)
- Read [our paper](https://peerj.com/articles/453/) (or [this other paper, for skimage in microtomography](https://ascimaging.springeropen.com/articles/10.1186/s40679-016-0031-0))

```{code-cell} ipython3

```
