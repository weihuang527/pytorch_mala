import h5py
import numpy as np
from scipy import ndimage
# import matplotlib.pyplot as plt

class GrowBoundary(object):
    '''Grow a boundary between regions in a label array. Does not grow at the
    border of the batch or an optionally provided mask.

    Args:

        labels (:class:`ArrayKey`):

            The array containing labels.

        mask (:class:`ArrayKey`, optional):

            A mask indicating unknown regions. This is to avoid boundaries to
            grow between labelled and unknown regions.

        steps (``int``, optional):

            Number of voxels (not world units!) to grow.

        background (``int``, optional):

            The label to assign to the boundary voxels.

        only_xy (``bool``, optional):

            Do not grow a boundary in the z direction.
    '''

    def __init__(self, labels, mask=None, steps=1, background=0, only_xy=False):
        self.labels = labels
        self.mask = mask
        self.steps = steps
        self.background = background
        self.only_xy = only_xy

    def process(self):

        gt = self.labels
        gt_mask = self.mask

        if gt_mask is not None:
            
            pass

        else:

            boundary = self.__grow(gt, only_xy=self.only_xy)
            return boundary

    def __grow(self, gt, gt_mask=None, only_xy=False):
        if gt_mask is not None:
            assert gt.shape == gt_mask.shape, "GT_LABELS and GT_MASK do not have the same size."

        if only_xy:
            boundary = np.zeros(shape=gt.shape)
            assert len(gt.shape) == 3
            for z in range(gt.shape[0]):
                boundary[z] = self.__grow(gt[z], None if gt_mask is None else gt_mask[z])
            return boundary

        # get all foreground voxels by erosion of each component
        foreground = np.zeros(shape=gt.shape, dtype=np.bool)
        masked = None
        if gt_mask is not None:
            masked = np.equal(gt_mask, 0)
        for label in np.unique(gt):
            if label == self.background:
                continue
            label_mask = gt==label
            # Assume that masked out values are the same as the label we are
            # eroding in this iteration. This ensures that at the boundary to
            # a masked region the value blob is not shrinking.
            if masked is not None:
                label_mask = np.logical_or(label_mask, masked)
            eroded_label_mask = ndimage.binary_erosion(label_mask, iterations=self.steps, border_value=1)
            foreground = np.logical_or(eroded_label_mask, foreground)

        # plt.figure()
        # plt.imshow(foreground, cmap='gray')
        # plt.show()
        # label new background
        background = np.logical_not(foreground)
        gt[background] = self.background
        return gt

if __name__ == "__main__":
    input_path = '../data/FIB_segmentaion/groundtruth.h5'
    dataset = 'stack'
    with h5py.File(input_path,'r') as f:
        label = f[dataset][:]
    
    label = label[:500, :500, :500]

    growboundary = GrowBoundary(label, steps=3, only_xy=True)
    gen = growboundary.process()

    print('Done')