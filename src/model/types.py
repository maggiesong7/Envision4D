from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch view gaussian 3"]
    scales: Float[Tensor, "batch view gaussian 3"]
    rotations: Float[Tensor, "batch view gaussian 4"]
    colors: Float[Tensor, "batch view gaussian 3"]
    opacities: Float[Tensor, "batch view gaussian"]
    motions: Float[Tensor, "batch view t gaussian 3"]
