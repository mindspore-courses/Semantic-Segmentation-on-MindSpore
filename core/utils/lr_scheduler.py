import math
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


# Base LRScheduler class
class LRScheduler(LearningRateSchedule):
    def __init__(self, mode, base_lr=0.01, target_lr=0, niters=0, nepochs=0, iters_per_epoch=0,
                 offset=0, power=0.9, step_iter=None, step_epoch=None, step_factor=0.1, warmup_epochs=0):
        super(LRScheduler, self).__init__()
        assert (mode in ['constant', 'step', 'linear', 'poly', 'cosine'])

        if mode == 'step':
            assert (step_iter is not None or step_epoch is not None)
        self.niters = niters
        self.step = step_iter
        epoch_iters = nepochs * iters_per_epoch
        if epoch_iters > 0:
            self.niters = epoch_iters
            if step_epoch is not None:
                self.step = [s * iters_per_epoch for s in step_epoch]

        self.step_factor = step_factor
        self.base_lr = base_lr
        self.target_lr = base_lr if mode == 'constant' else target_lr
        self.offset = offset
        self.power = power
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.mode = mode

    def get_lr(self, cur_step):
        N = self.niters - 1
        T = cur_step - self.offset
        T = min(max(0, T), N)

        if self.mode == 'constant':
            factor = 0
        elif self.mode == 'linear':
            factor = 1 - T / N
        elif self.mode == 'poly':
            factor = pow(1 - T / N, self.power)
        elif self.mode == 'cosine':
            factor = (1 + math.cos(math.pi * T / N)) / 2
        elif self.mode == 'step':
            if self.step is not None:
                count = sum([1 for s in self.step if s <= T])
                factor = pow(self.step_factor, count)
            else:
                factor = 1
        else:
            raise NotImplementedError

        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = factor * 1.0 * T / self.warmup_iters

        if self.mode == 'step':
            return self.base_lr * factor
        else:
            return self.target_lr + (self.base_lr - self.target_lr) * factor

# WarmupMultiStepLR class
class WarmupMultiStepLR(LearningRateSchedule):
    def __init__(self, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500, warmup_method="linear"):
        super(WarmupMultiStepLR, self).__init__()
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a list of increasing integers. Got {}", milestones)
        if warmup_method not in ("constant", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup_method accepted got {}".format(warmup_method))

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

    def get_lr(self, cur_step):
        warmup_factor = 1
        if cur_step < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(cur_step) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, cur_step) for base_lr in self.base_lrs]

# WarmupPolyLR class
class WarmupPolyLR(LearningRateSchedule):
    def __init__(self, target_lr=0, max_iters=0, power=0.9, warmup_factor=1.0 / 3, warmup_iters=500, warmup_method='linear'):
        super(WarmupPolyLR, self).__init__()
        if warmup_method not in ("constant", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup_method accepted got {}".format(warmup_method))

        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

    def get_lr(self, cur_step):
        N = self.max_iters - self.warmup_iters
        T = cur_step - self.warmup_iters
        if cur_step < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(cur_step) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return [self.target_lr + (base_lr - self.target_lr) * warmup_factor for base_lr in self.base_lrs]

        factor = pow(1 - T / N, self.power)
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]

if __name__ == '__main__':
    import mindspore.nn as nn

    model = nn.Conv2d(16, 16, 3, 1, pad_mode='pad', padding=1)
    optimizer = nn.optim.Adam(model.trainable_params(), learning_rate=0.01)
    lr_scheduler = WarmupPolyLR(optimizer, warmup_iters=1000)
    print(lr_scheduler)