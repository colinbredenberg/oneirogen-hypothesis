from beyond_backprop.datamodules.image_classification import ImageClassificationDataModule

from .inaturalist import INaturalistDataModule, TargetType, Version2021
from torchvision import transforms as T
from torchvision.datasets import INaturalist
from torch.utils.data import Subset
import pytest
import sys

slow = pytest.mark.skipif("-vvv" not in sys.argv, reason="Slow. Only runs when -vvv is passed.")


@slow
@pytest.mark.parametrize("version", ["2021_train", "2021_train_mini", "2021_valid"])
@pytest.mark.parametrize(
    "target_type", ["full", "kingdom", "phylum", "class", "order", "family", "genus"]
)
def test_dataset_download_works(target_type: TargetType, version: Version2021):
    batch_size = 64
    datamodule = INaturalistDataModule(
        batch_size=batch_size,
        version=version,
        target_type=target_type,
        train_transforms=T.Compose(
            [
                T.RandomResizedCrop(224),
                T.ToTensor(),
            ]
        ),
    )
    if datamodule.target_type == "full":
        assert isinstance(datamodule, ImageClassificationDataModule)
    datamodule.prepare_data()
    datamodule.setup(None)

    # assert (
    #     datamodule.num_samples
    #     == expected_total - datamodule.num_classes * datamodule.num_images_per_val_class
    # )
    for loader_fn in [
        datamodule.train_dataloader,
        datamodule.val_dataloader,
        datamodule.test_dataloader,
    ]:
        loader = loader_fn()
        assert isinstance(datamodule.dataset_train, Subset)
        dataset = datamodule.dataset_train.dataset
        assert isinstance(dataset, INaturalist)
        # assert False, (len(dataset.all_categories), len(dataset.categories_map))

        all_labels = set()

        from tqdm.rich import tqdm_rich

        for i, (x, y) in enumerate(
            tqdm_rich(loader, unit_scale=loader.batch_size, unit="Samples")
        ):
            assert x.shape == (batch_size, 3, 224, 224)
            assert y.shape == (batch_size,)
            all_labels.update(y.tolist())

            if i > 100:
                break

        min(all_labels)
        max(all_labels)

        # assert False, (
        #     min_label,
        #     max_label,
        #     len(set(range(min_label, max_label + 1)) - all_labels),
        #     list(itertools.islice(all_labels, 10)),
        # )
