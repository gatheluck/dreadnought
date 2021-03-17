import abc
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Optional, cast

import cleverhans.torch.attacks.fast_gradient_method  # type: ignore # noqa
import cleverhans.torch.attacks.projected_gradient_descent  # type: ignore # noqa
from omegaconf import MISSING


@dataclass
class AttackerConfig(abc.ABC):
    _target_: str = MISSING


@dataclass
class FgsmConfig(AttackerConfig):
    _target_: str = "cleverhans.torch.attacks.fast_gradient_method.fast_gradient_method"
    eps: float = MISSING
    norm: str = MISSING
    targeted: bool = MISSING


@dataclass
class PgdConfig(AttackerConfig):
    _target_: str = (
        "cleverhans.torch.attacks.projected_gradient_descent.projected_gradient_descent"
    )
    eps: float = MISSING
    eps_iter: Optional[float] = None
    nb_iter: int = MISSING
    norm: str = MISSING
    targeted: bool = MISSING
    rand_init: bool = MISSING


class Attacker(NamedTuple):
    name: str
    config: AttackerConfig


class ATTACKERS(Attacker, Enum):
    FGSM = Attacker("fgsm", cast(FgsmConfig, AttackerConfig))
    PGD = Attacker("pgd", cast(PgdConfig, AttackerConfig))
