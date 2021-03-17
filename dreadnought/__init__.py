import abc
from dataclasses import dataclass
from typing import Optional

import cleverhans.torch.attacks.projected_gradient_descent  # type: ignore # noqa
from omegaconf import MISSING


@dataclass
class AttackerConfig(abc.ABC):
    _target_: str = MISSING


@dataclass
class PgdConfig(AttackerConfig):
    _target_: str = "cleverhans.torch.attacks.projected_gradient_descent.projected_gradient_descent"
    eps: float = MISSING
    eps_iter: Optional[float] = None
    nb_iter: int = MISSING
    norm: str = MISSING
    targeted: bool = MISSING
    rand_init: bool = MISSING
