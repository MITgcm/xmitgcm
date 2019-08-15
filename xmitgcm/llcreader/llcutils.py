"""
Weird one-off functions that don't have a good home.
"""
import xarray
import numpy as np

def load_masks_from_mds(grid_dir):
    import xmitgcm
    ds = xmitgcm.open_mdsdataset(grid_dir, iters=None, geometry='llc')

    points = ['W', 'W', 'S']
    masks = [ds['hfac' + point].reset_coords(drop=True).rename('mask' + point)
             for point in points]
    ds_mask = xr.merge(masks)
    for c in ds_mask.coords:
        ds_mask[c] = ds_mask[c].astype('i2')
    return ds_mask.chunk({'k': 1, 'face': -1})


def write_masks_to_zarr(ds_mask, output_path):
    import lzma
    lzma_filters = [dict(id=lzma.FILTER_DELTA, dist=1),
                    dict(id=lzma.FILTER_LZMA2, preset=1)]
    from numcodecs import LZMA
    compressor = LZMA(filters=lzma_filters, format=lzma.FORMAT_RAW)
    encoding = {vname: {'compressor': compressor} for vname in ds_mask.data_vars}
    return ds_mask.to_zarr(output_path, encoding=encoding)


def copy_masks_from_mds_to_zarr(grid_dir, output_path):
    ds_mask = load_masks_from_mds(grid_dir)
    write_masks_to_zarr(ds_mask, output_path)


_facet_strides = ((0,3), (3,6), (6,7), (7,10), (10,13))
def face_mask_to_facet_index_list(mask):
    nk, nf, nx, ny = mask.shape
    assert nf == 13
    index = np.asarray(mask.sum(axis=(2, 3)))
    index_facets = np.array([list(index[:, slice(*stride)].sum(axis=1))
                             for stride in _facet_strides]).transpose()
    return [0] + list(index_facets.ravel().cumsum())
