from hydra.core.config_store import ConfigStore

from .algorithm import Algorithm
from .backprop import Backprop
from .dtp import DTP
from .reinforce.reinforce import REINFORCE
from .reinforce.rl_models import RLModel
from .wake_sleep.wake_sleep_layered_models import WSLayeredModel
from .wake_sleep.wake_sleep_layered_models import RMWSLayeredModel
from .wake_sleep.wake_sleep_layered_models import SingleLayeredModel
from .wake_sleep.wake_sleep_layered_models import PCAModel
from .wake_sleep.wake_sleep_layered_models import NormalizationModel
from .wake_sleep.wake_sleep_layered_models import FCWSLayeredModel
from .wake_sleep.wake_sleep_layered_models import MaskedFCWSLayeredModel
from .wake_sleep.wake_sleep_layered_models import SteerablePyramidModel
from .wake_sleep.wake_sleep_layered_models import SteerablePyramidConvModel
from .wake_sleep.wake_sleep_layered_models import TinyAutoencoderFCNet
from .wake_sleep.wake_sleep_layered_models import FCDiffusionModel
from .wake_sleep.wake_sleep_layered_models import LayerwiseDiffusionModel
from .wake_sleep.wake_sleep_layered_models import ConvDiffusionModel
from .wake_sleep.wake_sleep_layered_models import FCRecurrentModel

from .wake_sleep.bp_wake_models import FCGenNet
from .wake_sleep.bp_wake import BPWake
from .wake_sleep.vae import VAE
from .wake_sleep.vae_models import VAEFCGenNet
from .wake_sleep.vae_models import LadderVAEFCGenNet
from .wake_sleep.temporal_ws_models import TemporalLayeredModel
from .wake_sleep.wake_sleep import WakeSleep
from .wake_sleep.rm_wake_sleep import RMWakeSleep
from .wake_sleep.temporal_wake_sleep import TemporalWakeSleep
from .feedback_alignment.fa_network import FALayeredModel
from .feedback_alignment.feedback_alignment import FeedbackAlignment
from .example.example_algorithm import ExampleAlgorithm
from .image_classification import ImageClassificationAlgorithm
from .manual_optimization_example import ManualGradientsExample
from .rl_example.reinforce import ExampleRLAlgorithm
from .rl_example.dag_reinforce import DAGREINFORCE
from .rl_example.dag_reinforce_layered_models import DAGRLModel

# Store the different configuration options for each algorithm.

# NOTE: This works the same way as creating config files for each algorithm under
# `configs/algorithm`. From the command-line, you can select both configs that are yaml files as
# well as structured config (dataclasses).

# If you add a configuration file under `configs/algorithm`, it will also be available as an option
# from the command-line, and be validated against the schema.

_cs = ConfigStore.instance()
# _cs.store(group="algorithm", name="algorithm", node=Algorithm.HParams())
_cs.store(group="algorithm", name="backprop", node=Backprop.HParams())
_cs.store(group="algorithm", name="dtp", node=DTP.HParams())
_cs.store(group="algorithm", name="example", node=ExampleAlgorithm.HParams())
_cs.store(group="algorithm", name="manual_optimization", node=ManualGradientsExample.HParams())
_cs.store(group="algorithm", name="reinforce", node = REINFORCE.HParams())
_cs.store(group="algorithm", name = "wake_sleep", node = WakeSleep.HParams())
_cs.store(group="algorithm", name = "rm_wake_sleep", node = RMWakeSleep.HParams())
_cs.store(group = "algorithm", name = "feedback_alignment", node = FeedbackAlignment.HParams())
_cs.store(group="network", name="rlmodel", node=RLModel.HParams())
_cs.store(group = "network", name = "wsmodel", node = WSLayeredModel.HParams())
_cs.store(group = "network", name = "rmwsmodel", node = RMWSLayeredModel.HParams())
_cs.store(group = "network", name = "slmodel", node = SingleLayeredModel.HParams())
_cs.store(group= "network", name = "normmodel", node = NormalizationModel.HParams())
_cs.store(group= "network", name = "fcwsmodel", node = FCWSLayeredModel.HParams())
_cs.store(group= "network", name = "masked_fcwsmodel", node = MaskedFCWSLayeredModel.HParams())
_cs.store(group= "network", name = "pyramid_fcwsmodel", node = SteerablePyramidModel.HParams())
_cs.store(group= "network", name = "pyramid_conv_model", node = SteerablePyramidConvModel.HParams())
_cs.store(group= "network", name = "tinyac_fcnet", node = TinyAutoencoderFCNet.HParams())
_cs.store(group= "network", name = "diffusion_fcnet", node = FCDiffusionModel.HParams())
_cs.store(group= "network", name = "diffusion_conv", node = ConvDiffusionModel.HParams())
_cs.store(group= "network", name = "diffusion_layerwise", node = LayerwiseDiffusionModel.HParams())
_cs.store(group= "network", name = "recurrent_fcnet", node = FCRecurrentModel.HParams())
_cs.store(group= "network", name = "fcgennet", node = FCGenNet.HParams())
_cs.store(group= "network", name = "noiselessfcgennet", node = FCGenNet.HParams())
_cs.store(group = "network", name = "pcamodel", node = PCAModel.HParams())
_cs.store(group = "network", name = "dagrlmodel", node = DAGRLModel.HParams())
_cs.store(group = "network", name = "famodel", node = FALayeredModel.HParams())
_cs.store(group = "network", name = "vaefcnet", node = VAEFCGenNet.HParams())
_cs.store(group = "network", name = "ladder_vaefcnet", node = LadderVAEFCGenNet.HParams())
_cs.store(group = "network", name = "temporalwsmodel", node = TemporalLayeredModel.HParams())
_cs.store(group="algorithm", name="rl_example", node=ExampleRLAlgorithm.HParams())
_cs.store(group="algorithm", name="dag_reinforce", node = DAGREINFORCE.HParams())
_cs.store(group="algorithm", name="temporal_wake_sleep", node= TemporalWakeSleep.HParams())
_cs.store(group="algorithm", name="bp_wake", node = BPWake.HParams())
_cs.store(group="algorithm", name="vae", node = VAE.HParams())

__all__ = [
    "Algorithm",
    "Backprop",
    "DTP",
    "ExampleAlgorithm",
    "ExampleRLAlgorithm",
    "ImageClassificationAlgorithm",
    "ManualGradientsExample",
]
