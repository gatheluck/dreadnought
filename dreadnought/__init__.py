import abc
from dataclasses import dataclass
from typing import Optional

from cleverhans.torch.attacks.projected_gradient_descent import (  # type: ignore # noqa
    projected_gradient_descent,
)
from omegaconf import MISSING


@dataclass
class AttackerConfig(abc.ABC):
    _target_: str = MISSING


@dataclass
class PgdConfig(AttackerConfig):
    _target_: str = "projected_gradient_descent"
    eps: float = MISSING
    eps_iter: Optional[float] = None
    nb_iter: int = MISSING
    norm: str = MISSING
    targeted: bool = MISSING
    rand_init: bool = MISSING
