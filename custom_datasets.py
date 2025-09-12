import os.path as osp
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class UDD5Dataset(BaseSegDataset):
    """UDD5 dataset.
    
    """
    METAINFO = dict(
    classes=('vegetation', 'building', 'road', 'vehicle',
             'other'),
    palette=[[107, 142, 35], [102,102,156], [128,64,128],
             [0, 0, 142], [0, 0, 0]])

    def __init__(self,
                 img_suffix='.JPG',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)









