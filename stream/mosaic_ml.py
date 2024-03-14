import os
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from time import time
from utils import to_rgb
import torchvision.transforms.v2 as T
from tqdm import tqdm
from lightning_cloud.utils.data_connection import add_s3_connection
from lightning import seed_everything
#from lightning.data.streaming.resolver import _resolve_dir
import torch
import shutil
from typing import Any, Callable, Optional, Tuple

# 1. Clean cache
cache_dir = f"/tmp/mosaic_ml_imagenet_{time()}"
#if os.path.isdir(cache_dir):
#    shutil.rmtree(cache_dir)
os.makedirs(cache_dir, exist_ok=True)

# 2. Fixed the seed across packages
seed_everything(42)

# 3. Get the optimized dataset
#add_s3_connection("optimized-imagenet-1m")


# 4. Create streaming dataset
class LitDataImageNetStreamingDataset(StreamingDataset):
    def __init__(self, local, remote):
        super().__init__(local=local, remote=remote, shuffle_block_size=1<<18, partition_algo='orig', shuffle=False, shuffle_seed=9176, shuffle_algo='py1s', batch_size=256)

        self.transform = T.Compose(
            [   
                T.RandomResizedCrop(224, antialias=True),
                T.RandomHorizontalFlip(),
                T.ToImage(),
                T.ToDtype(torch.float16, scale=True),
            ]
        )

    def __getitem__(self, index):
        obj = super().__getitem__(index)
        return self.transform(to_rgb(obj['x']))[:3], obj["y"]

class ImageNetStreamingDataset(StreamingDataset):
    def __init__(self,
                 *,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 epoch_size: Optional[int] = None,
                 predownload: Optional[int] = None,
                 cache_limit: Optional[int] = None,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1s',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: int = 1 << 18,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        StreamingDataset.__init__(self,
                                  remote=remote,
                                  local=local,
                                  split=split,
                                  download_retry=download_retry,
                                  download_timeout=download_timeout,
                                  validate_hash=validate_hash,
                                  keep_zip=keep_zip,
                                  epoch_size=epoch_size,
                                  predownload=predownload,
                                  cache_limit=cache_limit,
                                  partition_algo=partition_algo,
                                  num_canonical_nodes=num_canonical_nodes,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  shuffle_algo=shuffle_algo,
                                  shuffle_seed=shuffle_seed,
                                  shuffle_block_size=shuffle_block_size)
        self.transform = T.Compose(
            [   
                T.RandomResizedCrop(224, antialias=True),
                T.RandomHorizontalFlip(),
                T.ToImage(),
                T.ToDtype(torch.float16, scale=True),
            ]
        )

    def __getitem__(self, index):
        obj = super().__getitem__(index)
        return self.transform(to_rgb(obj['x']))[:3], obj["y"]


# 5. Define the dataset and dataLoader
#dataset = ImageNetStreamingDataset(local=cache_dir, remote = "s3://mosaicml-internal-regression-test/train", batch_size=256)
dataset = LitDataImageNetStreamingDataset(local=cache_dir, remote = "s3://mosaicml-internal-regression-test/train")
dataloader = DataLoader(
        dataset, 
        batch_size=256,
        num_workers= os.cpu_count(),
)

# 6. Iterate over the datasets for 2 epochs
for epoch in range(2):
    num_samples = 0
    t0 = time()
    for data in tqdm(dataloader, smoothing=0, mininterval=1):
        num_samples += data[0].squeeze(0).shape[0]
    print(f'For {__file__} on {epoch}, streamed over {num_samples} samples in {time() - t0} or {num_samples / (time() - t0)} images/sec.')

# 7. Cleanup cache
#shutil.rmtree(cache_dir)
