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

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt

from skimage import data
camera = data.camera()

import numpy as np
```

<span style="font-size: 150%;">**Exercise:** Write an algorithm from scratch that will
take an input image as an ndarray, and rotate it by $\theta$ degrees.</span>

If you feel creative, you can also write code to magnify (zoom) the image.
<p></p>
You may need: http://en.wikipedia.org/wiki/Polar_coordinate_system
<p></p>
A (bad) solution is given below--but try it yourself before looking!

```{code-cell} ipython3
def interp(x, y, vals):
    """Bilinear interpolation.
    
    Parameters
    ----------
    x, y : float
        Position of required value inside a 1x1 pixel.
    vals : list of 4 floats
        Values on the four corners of the pixels, in order
        top left, top right, bottom left, bottom right.
        
    Returns
    -------
    v : float
        Interpolated value at position ``(x, y)``.
    
    """
    top_left, top_right, bottom_left, bottom_right = vals
    
    top_middle = (1 - x) * top_left + x * top_right
    bottom_middle = (1 - x) * bottom_left + x * bottom_right
    
    return (1 - y) * top_middle + y * bottom_middle


def rotate(image, theta):
    theta = np.deg2rad(theta)
    
    height, width = image.shape[:2]
    out = np.zeros_like(image)
    
    centre_x, centre_y = width / 2., height / 2.
    
    for x in range(width):
        for y in range(height):
            
            x_c = x - centre_x
            y_c = y - centre_y
            
            # Determine polar coordinate of pixel
            radius = np.sqrt(x_c**2 + y_c**2)
            angle = np.arctan2(y_c, x_c)
            
            new_angle = angle - theta
            
            old_x = radius * np.cos(new_angle)
            old_y = radius * np.sin(new_angle)
            
            old_x = old_x + centre_x
            old_y = old_y + centre_y
            
            if (old_x >= width - 1) or (old_x < 1) or\
               (old_y >= height - 1) or (old_y < 1):
                    continue
            else:
                xx = int(np.floor(old_x))
                yy = int(np.floor(old_y))
                
                out[y, x] = interp(old_x - xx, old_y - yy,
                                   [image[yy, xx], image[yy, xx + 1],
                                    image[yy + 1, xx], image[yy + 1, xx + 1]])
    
    return out

rotated = rotate(camera, 40)
    
plt.imshow(rotated, cmap='gray', interpolation='nearest');
```

# Advanced challenge: rectifying an image

<img src="../../images/chapel_floor.png" style="float: left;"/>

+++

Rectify the tiles above.

```{code-cell} ipython3
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from skimage import transform


from skimage.transform import estimate_transform

source = np.array([(129, 72),
                   (302, 76),
                   (90, 185),
                   (326, 193)])

target = np.array([[0, 0],
                   [400, 0],
                   [0, 400],
                   [400, 400]])

tf = estimate_transform('projective', source, target)
H = tf.params   # in older versions of skimage, this should be
                # H = tf._matrix

print(H)

# H = np.array([[  3.04026872e+00,   1.04929628e+00,  -4.67743998e+02],
#               [ -1.44134582e-01,   6.23382067e+00,  -4.30241727e+02],
#               [  2.63620673e-05,   4.17694527e-03,   1.00000000e+00]])

def rectify(xy):
    x = xy[:, 0]
    y = xy[:, 1]

    # You must fill in your code here.
    #
    # Handy functions are:
    #
    # - np.dot (matrix multiplication)
    # - np.ones_like (make an array of ones the same shape as another array)
    # - np.column_stack
    # - A.T -- type .T after a matrix to transpose it
    # - x.reshape -- reshapes the array x

    # We need to provide the backward mapping
    HH = np.linalg.inv(H)

    homogeneous_coordinates = np.column_stack([x, y, np.ones_like(x)])
    xyz = np.dot(HH, homogeneous_coordinates.T)

    # We want one coordinate per row
    xyz = xyz.T

    # Turn z into a column vector
    z = xyz[:, 2]
    z = z.reshape([len(z), 1])

    xyz = xyz / z

    return xyz[:, :2]

image = plt.imread('../../images/chapel_floor.png')
out = transform.warp(image, rectify, output_shape=(400, 400))

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
ax0.imshow(image)
ax1.imshow(out)

plt.show()
```

```{code-cell} ipython3

```
