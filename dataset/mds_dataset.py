import json
import os
import warnings
from typing import Any, List, Optional, Union, Sequence, Iterator, Tuple, Callable, Dict


import torch
from torch.utils.data import DataLoader, default_collate

from streaming.base.constant import TICK
from streaming.base.format import Reader, reader_from_json
from streaming import Stream, StreamingDataset
from streaming.base.util import wait_for_file_to_exist
from streaming.base.world import World

from .constants import BatchKeys
from .dataset import ProcessedDataset, DEFAULT_DATA_AUG_TARGETS
from .dataset import logger
from .mds_patches import patch_mds_encoding

patch_mds_encoding()

def get_nb_samples_in_stream(index_file: str) -> int:
    total_samples = 0
    with open(index_file, "r") as f:
        index = json.load(f)
    for shard in index["shards"]:
        total_samples += shard["samples"]
    return total_samples


def get_split_folders(path: str, file_name: str = "index.json") -> List[str]:
    """Get list of folders containing the specified index file.

    Args:
        path: Local filesystem path to search
        file_name: Name of the index file to look for (default: "index.json")

    Returns:
        List of folder paths containing the index file
    """
    if os.path.isfile(os.path.join(path, file_name)):
        # Index file exists in the root path
        return [path]

    # Check if the index file exists in any subfolder
    sub_folders = []
    for sub_folder in [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]:
        if os.path.isfile(os.path.join(path, sub_folder, file_name)):
            sub_folders.append(os.path.join(path, sub_folder))

    return sub_folders


def get_stream_iterator(
    local: Union[str, List[str]],
    remote: Optional[Union[str, List[str]]],
    proportions: Optional[Union[float, List[float]]],
) -> Iterator[Tuple[Optional[str], str, str, Optional[float]]]:
    """Get iterator over (remote, local, index_file, proportion) tuples."""
    if remote is None:
        return get_local_iterator(local, proportions)

    if proportions is not None:
        warnings.warn("Proportions are ignored when using remote datasets")
    return get_remote_iterator(remote, local)


def get_local_iterator(
    local_paths: Union[str, List[str]],
    proportions: Optional[Union[float, List[float]]]
) -> Iterator[Tuple[Optional[str], str, str, Optional[float]]]:
    """Iterate over local dataset paths."""
    if isinstance(local_paths, str):
        local_paths = [local_paths]

    for idx, local_path in enumerate(local_paths):
        # Get proportion for this path
        if isinstance(proportions, Sequence):
            proportion = proportions[idx]
        else:
            proportion = proportions

        # Check for index files in root
        index_files = [f for f in os.listdir(local_path) if f.endswith("_index.json")]

        if index_files:
            # Multiple streams in root - use cache workaround
            logger.info(
                "Multiple streams in root directory. Using remote/local cache workaround."
            )
            tmp_path = local_path.rstrip("/").split("/")[-1]
            stream_props = split_proportion(proportion, index_files) if proportion else [None] * len(index_files)

            for i, (index_file, prop) in enumerate(zip(index_files, stream_props)):
                yield local_path, f"/tmp/{tmp_path}_{i}/", index_file, prop
        else:
            # Check subfolders
            folders = get_split_folders(local_path, "index.json")
            stream_props = split_proportion(proportion, folders) if proportion else [None] * len(folders)

            for folder, prop in zip(folders, stream_props):
                yield None, folder, "index.json", prop


def get_remote_iterator(
    remote_paths: Union[str, List[str]],
    local_paths: Union[str, List[str]]
) -> Iterator[Tuple[str, str, str, None]]:
    """Iterate over remote dataset paths with local cache."""
    # Normalize to lists
    if isinstance(remote_paths, str) and isinstance(local_paths, str):
        remote_paths, local_paths = [remote_paths], [local_paths]

    # Handle length mismatch
    if isinstance(remote_paths, Sequence) and isinstance(local_paths, Sequence):
        if len(remote_paths) != len(local_paths):
            if len(remote_paths) < len(local_paths):
                raise ValueError("More local paths than remote paths")

            # Create cache subfolders
            local_paths = list(local_paths)
            base_local = local_paths[-1]
            local_paths[-1] = f"{base_local}/cache_0"

            for i in range(len(remote_paths) - len(local_paths)):
                local_paths.append(f"{base_local}/cache_{i + 1}")

            warnings.warn(f"Created cache subfolders: {local_paths}")

            # Ensure directories exist
            for path in local_paths:
                os.makedirs(path, exist_ok=True)

    # Iterate over pairs
    for remote_path, local_path in zip(remote_paths, local_paths):
        remote_folders = get_split_folders(remote_path, "index.json")
        local_folders = [os.path.join(local_path, str(i)) for i in range(len(remote_folders))]

        for remote, local in zip(remote_folders, local_folders):
            yield remote, local, "index.json", None


def split_proportion(proportion: Optional[float], items: List[Any]) -> List[Optional[float]]:
    """Split a proportion across multiple items based on their sample counts."""
    if proportion is None:
        return [None] * len(items)

    # Get sample counts
    counts = [get_nb_samples_in_stream(item) for item in items]
    total = sum(counts)

    # Distribute proportion
    return [(count / total) * proportion for count in counts]


class PatchedStream(Stream):
    """
    Inherit form Stream.
    Update the get_shards method to take any index file not only index.json
    """

    def __init__(self, index_file: str = "index.json", *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.index_file = index_file

    def get_shards(self, world: World, allow_unsafe_types: bool) -> List[Reader]:
        """Load this Stream's index, retrieving its shard readers.

        Args:
            world (World): Distributed context.
            allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
                execution during deserialization, whether to keep going if ``True`` or raise an
                error.

        Returns:
            `List[Reader]: Shard readers.
        """
        basename = self.index_file
        filename = os.path.join(self.local, self.split, basename)  # pyright: ignore
        if not os.path.exists(filename):
            if world.is_local_leader:
                if self.remote:
                    # Downloads the `index.json` as `index.json.tmp` fully and then rename it to
                    # `index.json` since only one process downloads the `index.json` file while
                    # other processes wait for it to get downloaded. Hence, It avoids loading the
                    # in-progress downloading `index.json`.
                    tmp_filename = self._download_file(basename, basename + ".tmp")
                    os.rename(tmp_filename, filename)
                else:
                    if not os.path.exists(filename):
                        raise RuntimeError(
                            f"No `remote` provided, but local file {filename} " + "does not exist either"
                        )
            else:
                wait_for_file_to_exist(
                    filename,
                    TICK,
                    self.download_timeout,
                    f"Index file {os.path.join(self.remote or '', self.split or '', basename)} "
                    + f"-> {filename} took too long to download or failed to download. Either increase the "
                    + "`download_timeout` value or check the local rank 0 traceback.",
                )
        try:
            obj = json.load(open(filename))
        except json.decoder.JSONDecodeError as error:
            error.args = (f"Index file at {filename} is empty or corrupted. " + error.args[0],)
            raise error

        # Version check.
        if obj["version"] != 2:
            raise ValueError(f"Unsupported streaming data version: {obj['version']}. Expected version 2.")

        # Initialize shard readers according to the loaded info.
        shards = []
        for info in obj["shards"]:
            shard = reader_from_json(self.local, self.split, info)
            shard.validate(allow_unsafe_types)
            shards.append(shard)

        return shards

    def get_index_size(self) -> int:
        """Get the size of the index file in bytes.

        Returns:
            int: Size in bytes.
        """
        filename = os.path.join(self.local, self.split, self.index_file)
        return os.stat(filename).st_size


class StreamingProcessedDataset(StreamingDataset, ProcessedDataset):
    """Streaming dataset implementation."""

    def __init__(
        self,
        streams: Optional[Sequence[Stream]] = None,
        caption_keys: Union[str, List[str], List[Tuple[str, float]]] = "caption",
        text_tower: str = "t5gemma2b-256-bf16",
        split: Optional[str] = None,
        download_retry: Optional[int] = 500,
        download_timeout: Optional[float] = 120,
        predownload: Optional[int] = None,
        cache_limit: Optional[str] = "1tb",
        num_canonical_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        shuffle_seed: int = 9146,
        has_text_latents: bool = True,
        has_mask_text_latents: bool = True,
        batching_method: str = "per_stream",
        transforms: Optional[List[Callable]] = None,
        transforms_targets: Union[List[str], str] = DEFAULT_DATA_AUG_TARGETS,
    ):
        StreamingDataset.__init__(
            self,
            streams=streams,
            remote=None,
            local=None,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=None,
            keep_zip=False,
            predownload=predownload,
            cache_limit=cache_limit,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            batching_method=batching_method,
        )
        ProcessedDataset.__init__(
            self,
            caption_keys=caption_keys,
            text_tower=text_tower,
            has_text_latents=has_text_latents,
            has_mask_text_latents=has_mask_text_latents,
            transforms=transforms,
            transforms_targets=transforms_targets,
        )

    def __getitem__(self, index: int) -> Optional[Dict[BatchKeys, Any]]:
        return ProcessedDataset.__getitem__(self, index)

    def _get_raw_item(self, index: int) -> Dict[str, Any]:
        return StreamingDataset.__getitem__(self, index)


def build_streaming_processed_dataloader(
    local: Union[str, List[str]],
    batch_size: int,
    remote: Optional[Union[str, List[str]]] = None,
    caption_keys: Union[str, List[str], List[Tuple[str, float]]] = "caption",
    text_tower: str = "t5gemma2b-256-bf16",
    num_samples: Optional[int] = None,
    cache_limit: str = "1tb",
    predownload: Optional[int] = None,
    download_retry: int = 2,
    download_timeout: float = 120,
    drop_last: bool = True,
    shuffle: bool = False,
    shuffle_seed: int = 9146,
    num_canonical_nodes: Optional[int] = None,
    proportions: Optional[Union[float, List[float]]] = None,
    batching_method: str = "per_stream",
    transforms: Optional[List[Callable]] = None,
    transforms_targets: Union[List[str], str] = DEFAULT_DATA_AUG_TARGETS,
    has_text_latents: bool = True,
    has_mask_text_latents: bool = False,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """Build a streaming dataloader for processed datasets."""

    # Create streams
    streams = []
    for r_path, l_path, index_file, proportion in get_stream_iterator(local, remote, proportions):
        if proportion is not None and proportion <= 0:
            raise ValueError(f"Proportion must be positive, got {proportion}")

        streams.append(
            PatchedStream(
                remote=r_path,
                local=l_path,
                download_retry=download_retry,
                download_timeout=download_timeout,
                index_file=index_file,
                proportion=proportion,
            )
        )

    # Build dataset
    dataset = StreamingProcessedDataset(
        streams=streams,
        caption_keys=caption_keys,
        text_tower=text_tower,
        split=None,
        download_retry=download_retry,
        download_timeout=download_timeout,
        predownload=predownload,
        cache_limit=cache_limit,
        num_canonical_nodes=num_canonical_nodes,
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        batching_method=batching_method,
        transforms=transforms,
        transforms_targets=transforms_targets,
        has_text_latents=has_text_latents,
        has_mask_text_latents=has_mask_text_latents,
    )

    # Apply subset if needed
    if num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(num_samples))

    # Log summary
    logger.info("--- Dataset Summary ---")
    logger.info(f"Remote: {remote} | Local: {local}")
    logger.info(f"Total size: {dataset.size} | This rank: {len(dataset)}")
    logger.info(f"Sum of stream samples: {sum(dataset.samples_per_stream)}")
    for i, (stream, n_samples) in enumerate(zip(dataset.streams, dataset.samples_per_stream)):
        location = stream.remote or stream.local
        index = getattr(stream, 'index_file', 'index.json')
        logger.info(f"  Stream {i}: {location}/{index} - {n_samples} samples")
    logger.info("-" * 23)

    # Build dataloader
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=drop_last,
        collate_fn=lambda batch: default_collate([x for x in batch if x is not None]),
        **dataloader_kwargs,
    )