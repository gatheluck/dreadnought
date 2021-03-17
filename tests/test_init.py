import numpy as np
from hydra.utils import instantiate

import dreadnought


class TestFgsm:
    def test__standard(
        self, pretrained_cifar10_resnet50, normalize_cifar10_loader, devices
    ):
        config = dreadnought.FgsmConfig
        config.eps = 8.0
        config.targeted = False
        config.rand_init = True
        norms = {np.inf, 1, 2}

        for norm in norms:
            for device in devices:
                model = pretrained_cifar10_resnet50.to(device)
                for x, t in normalize_cifar10_loader:
                    x, t = x.to(device), t.to(device)
                    x_adv = instantiate(config, model_fn=model, x=x, y=t, norm=norm)

                    assert not x.equal(x_adv)
                    break


class TestPgd:
    def test__standard(
        self, pretrained_cifar10_resnet50, normalize_cifar10_loader, devices
    ):
        config = dreadnought.PgdConfig
        config.eps = 8.0
        config.nb_iter = 10
        config.eps_iter = config.eps / config.nb_iter
        config.targeted = False
        config.rand_init = True
        norms = {np.inf, 2.0}

        for norm in norms:
            for device in devices:
                model = pretrained_cifar10_resnet50.to(device)
                for x, t in normalize_cifar10_loader:
                    x, t = x.to(device), t.to(device)
                    x_adv = instantiate(config, model_fn=model, x=x, norm=norm)

                    assert not x.equal(x_adv)
                    break
