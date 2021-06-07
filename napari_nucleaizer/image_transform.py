import numpy as np
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter
from skimage.transform import rescale

def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]

    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[0]
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect

def circularity(perimeter, area):
    """Calculate the circularity of the region

    Parameters
    ----------
    perimeter : float
        the perimeter of the region
    area : float
        the area of the region

    Returns
    -------
    circularity : float
        The circularity of the region as defined by 4*pi*area / perimeter^2
    """
    circularity = 4 * np.pi * area / (perimeter ** 2)

    return circularity


def draw_contours(im, mask):
    p = ((1,1), (1, 1))
    mask = np.pad(mask, p, mode='constant')

    for label in [u for u in np.unique(mask) if u > 0]:
        mask_lab = (mask==label).astype(np.uint8)

        ct_ = find_contours(mask_lab, .5)

        for ci in range(len(ct_)):
            ct = ct_[ci]
            rr, cc = polygon_perimeter(ct[:, 0], ct[:, 1], im.shape)
            im[rr, cc, 0] = 255

    im = im[1:-1, 1:-1]

    return im

def rescale_mask(mask, scale_factor):
    return rescale(mask, scale_factor, order=0, anti_aliasing=False, preserve_range=True)

def rescale_image(image, scale_factor):
    return rescale(image, scale_factor, anti_aliasing=True, preserve_range=True, multichannel=True).astype(np.uint8)
