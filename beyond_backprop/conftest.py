from __future__ import annotations

import typing
from typing import Any, Callable, ClassVar

import pytest
import torch
from lightning import seed_everything
from torch import Tensor, nn
from torch.utils.data import TensorDataset
from torchvision.datasets import VisionDataset

from beyond_backprop.configs.datamodule import VisionDataModuleConfig
from beyond_backprop.datamodules.vision_datamodule import VisionDataModule
from beyond_backprop.networks import LeNet, Network, ResNet18, ResNet34, SimpleVGG

if typing.TYPE_CHECKING:
    pass
num_classes: int = 10


@pytest.fixture(autouse=True)
def seed():
    """Fixture that seeds everything for reproducibility and yields the random seed used."""
    random_seed = 123
    seed_everything(random_seed, workers=True)
    yield random_seed


@pytest.fixture()
def x_y(seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 10
    num_channels = 3
    image_size: int = 32
    image_shape = [num_channels, image_size, image_size]
    gen = torch.Generator(device=device).manual_seed(seed)

    x = torch.rand([batch_size, *image_shape], generator=gen, device=device)
    y = torch.randint(0, num_classes, [batch_size], generator=gen, device=device)
    return x, y


@pytest.fixture(
    params=[
        SimpleVGG,
        LeNet,
        ResNet18,
        ResNet34,
    ]
)
def base_network(request: pytest.FixtureRequest, x_y: tuple[Tensor, Tensor]):
    """Fixture that creates a "base" network with the default hyper-parameters."""
    # NOTE: This network creation should be seeded properly, since it uses x_y which uses the
    # `seed` fixture.
    x, _ = x_y
    network_type: type[Network] = request.param
    network = network_type(in_channels=x.shape[1], n_classes=num_classes, hparams=None)
    assert isinstance(network, nn.Module)
    network = network.to(x.device)
    # Warm-up the network with a forward pass, to instantiate all the layer weights.
    _ = network(x)
    return network


class DummyDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        train: bool = True,
        download: bool = False,
    ):
        n_samples = 1000
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_features=4 * 4 * 3,
            n_repeated=4 * 4 * 2,
            n_informative=4 * 4 * 1,
            n_classes=10,
            n_clusters_per_class=1,
            n_samples=n_samples,
            random_state=123,
        )
        X = X.reshape(n_samples, 4, 4, 3)
        # FIXME: Make this look more like an image dataset by repeating fewer features:
        X2 = X.repeat(1, 8, 8, 1)
        assert False, (X.shape, X2.shape)

        X = X.reshape(-1, 3, 32, 32)
        # X *= 256
        # X = X.astype("uint8")
        self.data = TensorDataset(
            torch.as_tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.long)
        )
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )

    def __getitem__(self, index: Any):
        x, y = self.data[index]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def __len__(self) -> int:
        return len(self.data)


class DummyDataModule(VisionDataModule):
    dataset_cls: type[VisionDataset] = DummyDataset
    dims: ClassVar[tuple[int, int, int]] = (3, 32, 32)
    num_classes: ClassVar[int] = 10


@pytest.fixture(autouse=True, scope="session")
def add_dummy_datamodule_as_option():
    from hydra.core.config_store import ConfigStore
    from hydra_zen import builds
    from torchvision.transforms import Compose, Normalize

    from beyond_backprop.utils.hydra_utils import config_name

    cs = ConfigStore.instance()

    DummyDataModuleConfig = builds(
        DummyDataModule,
        train_transforms=builds(
            Compose, transforms=[builds(Normalize, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        ),
        builds_bases=(VisionDataModuleConfig,),
        zen_dataclass={"cls_name": config_name(DummyDataModule)},
    )
    # NOTE: This actually seems to be harder to learn for a ConvNet than CIFAR10 or MNIST!
    # This is probably because the "features" are not spacially correlated at all!
    cs.store(
        group="datamodule",
        name="dummy",
        node=DummyDataModuleConfig,
    )
