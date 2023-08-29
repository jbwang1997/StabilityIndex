import math
import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import Sampler as _Sampler
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .waymo.waymo_dataset import WaymoDataset
from .pandaset.pandaset_dataset import PandasetDataset
from .lyft.lyft_dataset import LyftDataset
from .once.once_dataset import ONCEDataset
# from .argo2.argo2_dataset import Argo2Dataset
from .custom.custom_dataset import CustomDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDataset': WaymoDataset,
    'PandasetDataset': PandasetDataset,
    'LyftDataset': LyftDataset,
    'ONCEDataset': ONCEDataset,
    'CustomDataset': CustomDataset,
    # 'Argo2Dataset': Argo2Dataset
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class DistributedSeqSampler(_DistributedSampler):
    """This sampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, max_interval=5):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.max_interval = max_interval
        assert hasattr(self.dataset, 'group_idx_by_seq')
    
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        seq_groups = self.dataset.group_idx_by_seq()
        intervals = torch.randint(
            1, self.max_interval+1, (len(seq_groups), ), generator=g).tolist()
        seq_pairs = []
        for seq_idx, interval in zip(seq_groups, intervals):
            if len(seq_idx) % (2 * interval) != 0:
                seq_idx += [seq_idx[-1]] * ((2 * interval) - len(seq_idx) % (2 * interval))
            seq_idx = torch.Tensor(seq_idx).reshape(-1, interval).permute(1, 0)
            seq_idx = seq_idx.reshape(-1, 2)
            seq_pairs.append(seq_idx[seq_idx[:, 0] != seq_idx[:, 1]])

        seq_pairs = torch.cat(seq_pairs, dim=0)
        seq_pairs = seq_pairs[torch.randperm(len(seq_pairs), generator=g)]

        if seq_pairs.numel() < self.total_size:
            padding_size = math.ceil(self.total_size / 2) - seq_pairs.shape[0]
            padding_pairs = seq_pairs[torch.randint(
                0, seq_pairs.shape[0], (padding_size, ), generator=g)]
            seq_pairs = torch.cat([seq_pairs, padding_pairs], dim=0)

        if seq_pairs.shape[0] % self.num_replicas != 0:
            padding_size = self.num_replicas - seq_pairs.shape[0] % self.num_replicas
            padding_pairs = seq_pairs[torch.randint(
                0, seq_pairs.shape[0], (padding_size, ), generator=g)]
            seq_pairs = torch.cat([seq_pairs, padding_pairs], dim=0)
        
        indices = seq_pairs[self.rank::self.num_replicas].long().reshape(-1).tolist()
        if len(indices) > self.num_samples:
            indices = indices[:self.num_samples]
        return iter(indices)


class SeqSampler(_Sampler):

    def __init__(self, dataset, max_interval=5):
        self.dataset = dataset
        self.max_interval = max_interval
        self.epoch = 0
        assert hasattr(self.dataset, 'group_idx_by_seq') 
    
    def __len__(self):
        return len(self.dataset)
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        seq_groups = self.dataset.group_idx_by_seq()
        intervals = torch.randint(
            1, self.max_interval+1, (len(seq_groups), ), generator=g).tolist()
        seq_pairs = []
        for seq_idx, interval in zip(seq_groups, intervals):
            if len(seq_idx) % (2 * interval) != 0:
                seq_idx += [seq_idx[-1]] * ((2 * interval) - len(seq_idx) % (2 * interval))
            seq_idx = torch.Tensor(seq_idx).reshape(-1, interval).permute(1, 0)
            seq_idx = seq_idx.reshape(-1, 2)
            seq_pairs.append(seq_idx[seq_idx[:, 0] != seq_idx[:, 1]])

        seq_pairs = torch.cat(seq_pairs, dim=0)
        seq_pairs = seq_pairs[torch.randperm(len(seq_pairs), generator=g)]

        if seq_pairs.numel() < len(self.dataset):
            padding_size = math.ceil(len(self.dataset) / 2) - seq_pairs.shape[0]
            padding_pairs = seq_pairs[torch.randint(
                0, seq_pairs.shape[0], (padding_size, ), generator=g)]
            seq_pairs = torch.cat([seq_pairs, padding_pairs], dim=0)

        indices = seq_pairs.reshape(-1).long().tolist()
        if len(indices) > len(self.dataset):
            indices = indices[:len(self.dataset)]
        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    seq_sample_cfg = dataset_cfg.get('SEQ_SAMPLE_CFG', None)
    if dist:
        rank, world_size = common_utils.get_dist_info()
        if training:
            if seq_sample_cfg is not None and seq_sample_cfg.ENABLE:
                assert batch_size % 2 == 0
                sampler = DistributedSeqSampler(dataset, world_size, rank, max_interval=seq_sample_cfg.MAX_INTERVAL)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        if training and seq_sample_cfg is not None and seq_sample_cfg.ENABLE:
            assert batch_size % 2 == 0
            sampler = SeqSampler(dataset, max_interval=seq_sample_cfg.MAX_INTERVAL)
        else:
            sampler = None

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )

    return dataset, dataloader, sampler
