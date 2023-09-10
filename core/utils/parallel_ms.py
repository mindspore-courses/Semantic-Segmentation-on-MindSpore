import threading
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn import Cell

# Reduce class
class Reduce(Cell):
    def construct(self, *inputs):
        return P.TensorAddN()(inputs)


# DataParallelModel class
class DataParallelModel(Cell):
    def __init__(self, module):
        super(DataParallelModel, self).__init__()
        self.module = module

    def construct(self, x):
        return self.module(x)

# DataParallelCriterion class
class DataParallelCriterion(Cell):
    def __init__(self, criterion):
        super(DataParallelCriterion, self).__init__()
        self.criterion = criterion

    def construct(self, inputs, targets):
        return self.criterion(inputs, targets)

# Get a variable from an object
def get_a_var(obj):
    if isinstance(obj, Tensor):
        return obj
    return None

# Apply criterion in parallel
def criterion_parallel_apply(modules, inputs, targets):
    lock = threading.Lock()
    results = {}

    def _worker(i, module, input, target):
        output = module(input, target)
        with lock:
            results[i] = output

    threads = [threading.Thread(target=_worker, args=(i, module, input, target))
               for i, (module, input, target) in enumerate(zip(modules, inputs, targets))]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    outputs = [results[i] for i in range(len(inputs))]
    return P.TensorAddN()(outputs) / len(outputs)
