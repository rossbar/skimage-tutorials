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

# StackOverflow Problems

+++

### Real-world problems to test your skills on!

```{code-cell} ipython3
%matplotlib inline
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from skimage import (filters, io, color, exposure, feature,
                     segmentation, morphology, img_as_float)
```

# Parameters of a pill

(Based on StackOverflow http://stackoverflow.com/questions/28281742/fitting-a-circle-to-a-binary-image)

<img src="../../images/round_pill.jpg" width="200px" style="float: left; padding-right: 1em;"/>
Consider a pill from the [NLM Pill Image Recognition Pilot](http://pir.nlm.nih.gov/pilot/instructions.html) (``../../images/round_pill.jpg``).  Fit a circle to the pill outline and compute its area.

<div style="clear: both;"></div>

*Hints:*

1. Equalize (``exposure.equalize_*``)
2. Detect edges (``filter.canny`` or ``feature.canny``--depending on your version)
3. Fit the ``CircleModel`` using ``measure.ransac``.

```{code-cell} ipython3
image = io.imread("../../images/round_pill.jpg")
image_equalized = exposure.equalize_adapthist(image)
edges = feature.canny(color.rgb2gray(image_equalized))

f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 8))
ax0.imshow(image)
ax1.imshow(image_equalized)
ax2.imshow(edges, cmap='gray');
```

```{code-cell} ipython3
from skimage import measure
from matplotlib.patches import Circle

coords = np.column_stack(np.nonzero(edges))

model, inliers = measure.ransac(coords, measure.CircleModel,
                                min_samples=3, residual_threshold=1,
                                max_trials=500)

print('Circle parameters:', model.params)

row, col, radius = model.params

f, ax = plt.subplots()
ax.imshow(image, cmap='gray');
circle = Circle((col, row), radius=radius, edgecolor='C9', linewidth=2, fill=False)
ax.add_artist(circle);
```

### Alternative: morphological snakes

**NOTE**: this is expensive to compute, so may take a while to execute

```{code-cell} ipython3
# Initial level set
pill = color.rgb2gray(image)
pill = restoration.denoise_nl_means(pill, multichannel=False)

level_set = segmentation.circle_level_set(pill.shape, radius=200)
ls = segmentation.morphological_chan_vese(pill, 80, init_level_set=level_set, smoothing=3)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ax.imshow(pill, cmap="gray")
ax.set_axis_off()
ax.contour(ls, [0.5], colors='r');
```

# Counting coins

Based on StackOverflow http://stackoverflow.com/questions/28242274/count-number-of-objects-using-watershed-algorithm-scikit-image

Consider the coins image from the scikit-image example dataset, shown below.
Write a function to count the number of coins.

The procedure outlined here is a bit simpler than in the notebook lecture (and works just fine!)

<div style="clear: both;"></div>

*Hint:*

1. Equalize
2. Threshold (``filters.threshold_otsu``)
3. Remove objects touching boundary (``segmentation.clear_border``)
4. Apply morphological closing (``morphology.closing``)
5. Remove small objects (``measure.regionprops``)
6. Visualize (potentially using ``color.label2rgb``)

```{code-cell} ipython3
from skimage import data
fig, ax = plt.subplots()
ax.imshow(data.coins(), cmap='gray');
```

```{code-cell} ipython3
from skimage import segmentation

image = data.coins()
equalized = exposure.equalize_adapthist(image)
binary0 = equalized > filters.threshold_otsu(equalized)
binary1 = segmentation.clear_border(binary0)
binary2 = morphology.closing(binary1, morphology.square(3))

f, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(image, cmap='gray')
ax1.imshow(edges, cmap='gray');
```

```{code-cell} ipython3
labels = ndi.label(binary2)[0]
labels_big = morphology.remove_small_objects(labels)
print("Number of coins:", len(np.unique(labels_big)[1:]))

out = color.label2rgb(labels_big, image, bg_label=0)
fig, ax = plt.subplots()
ax.imshow(out);
```

# Snakes

Based on https://stackoverflow.com/q/8686926/214686

<img src="../../images/snakes.png" width="200px" style="float: left; padding-right: 1em;"/>

Consider the zig-zaggy snakes on the left (``../../images/snakes.png``).<br/>Write some code to find the begin- and end-points of each.

<div style="clear: both;"></div>

*Hints:*

1. Threshold the image to turn it into "black and white"
2. Not all lines are a single pixel thick.  Use skeletonization to thin them out (``morphology.skeletonize``)
3. Locate all snake endpoints (I used a combination of ``scipy.signal.convolve2d`` [find all points with only one neighbor], and ``np.logical_and`` [which of those points lie on the snake?] to do that, but there are many other ways).

```{code-cell} ipython3
from scipy import ndimage as ndi

image = color.rgb2gray(io.imread('../../images/snakes.png'))

# Reduce all lines to one pixel thickness
snakes = morphology.skeletonize(image < 1).astype(np.uint8)

# Find pixels with only one neighbor
neighbor_kernel = np.array([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]])
num_neighbors = ndi.convolve(snakes, neighbor_kernel,
                             mode='constant')
corners = (num_neighbors == 1) & snakes

# Those are the start and end positions of the segments
rr, cc = np.nonzero(corners)

fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.scatter(cc, rr)
ax.set_axis_off()

plt.show()
```

# M&Ms

How many blue M&Ms are there in this image (`../../images/mm.jpg`)?

<img src="../../images/mm.jpg" width="400px"/>

Steps:

1. Denoise the image (using, e.g., `restoration.denoise_nl_means`)
2. Calculate how far each pixel is away from pure blue
3. Segment this distance map to give a "pill mask"
4. Fill in any holes in that mask, using `scipy.ndimage.binary_fill_holes`
5. Use watershed segmentation to split apart any M&Ms that were joined, as described in http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html

*Alternative approach:*

- http://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html

```{code-cell} ipython3
image = img_as_float(io.imread('../../images/mm.jpg'))
blurred = restoration.denoise_nl_means(image, multichannel=True)
blue_pill_color = [0, 0, 1]  # rough approximation; good enough

blue_dist = np.linalg.norm(blurred - [0, 0, 1], axis=2)
blue_mask = blue_dist < np.percentile(blue_dist, 5)
blue_mask = ndi.binary_fill_holes(blue_mask)

plt.imshow(blue_mask, cmap='gray');
```

```{code-cell} ipython3
distance = ndi.distance_transform_edt(blue_mask)
local_maxi = feature.peak_local_max(
    distance, indices=False, footprint=np.ones((5, 5)), labels=blue_mask
)
markers = ndi.label(local_maxi)[0]
labels = segmentation.watershed(-distance, markers, mask=blue_mask)

plt.imshow(labels, cmap='tab20');
```

```{code-cell} ipython3
print("There are {} blue M&M's".format(np.max(labels)))
```

# Viscous fingers

Based on StackOverflow: http://stackoverflow.com/questions/23121416/long-boundary-detection-in-a-noisy-image

<img src="../../images/fingers.png" width="200px" style="float: left; padding-right: 1em;"/>

Consider the fluid experiment on the right.  Determine any kind of meaningful boundary in this noisy image.

<div style="clear: both;"></div>

*Hints:*

1. Convert to grayscale
2. Try edge detection (``feature.canny``)
3. If edge detection fails, denoising is needed (try ``restoration.denoise_tv_bregman``)
4. Try edge detection (``feature.canny``)

```{code-cell} ipython3
from skimage import restoration, color, io, feature, morphology

image = color.rgb2gray(img_as_float(io.imread('../../images/fingers.png')))
denoised = restoration.denoise_nl_means(image, h=0.06, multichannel=False)
edges = feature.canny(denoised, low_threshold=0.001, high_threshold=0.75, sigma=1)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 15))
ax0.imshow(denoised, cmap='gray')
ax1.imshow(edges, cmap='gray', interpolation='lanczos')
for ax in (ax0, ax1):
    ax.set_axis_off()
```
