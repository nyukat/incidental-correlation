from datamodules.datamodule_base import BaseDataModule
from datasets.nlvr2_dataset import NLVR2Dataset


class NLVR2DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return NLVR2Dataset

    @property
    def dataset_name(self):
        return "nlvr2"
