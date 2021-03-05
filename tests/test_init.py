import numpy as np
import dreadnought
from hydra.utils import instantiate

from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent


class TestPgd:
    def test__standard(self, pretrained_cifar10_resnet50, normalize_cifar10_loader, devices):
        config = dreadnought.PgdConfig
        config.eps = 8.0
        config.nb_iter = 10
        config.eps_iter = config.eps / config.nb_iter
        config.targeted = False
        config.rand_int = True
        norms = {np.inf, 2.}

        for norm in norms:
            for device in devices:
                model = pretrained_cifar10_resnet50.to(device)
                for x, t in normalize_cifar10_loader:
                    x, t = x.to(device), t.to(device)
                    x_adv = instantiate(config, model_fn=model, x=x)
                    # x_adv = pgd.projected_gradient_descent(model, x, config.eps, config.eps_iter, config.nb_iter, norm, y=t)
                    break