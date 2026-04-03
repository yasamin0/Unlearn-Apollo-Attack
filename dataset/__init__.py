from .cifar10   import PartialCIFAR10
from .cifar100  import PartialCIFAR100
from .svhn      import PartialSVHN
from .toydata   import PartialToyData
from .partial_dataset import PartialDataset

def create_dataset(dataset_name, setting, root, img_size=32) -> PartialDataset:
    dataset = eval(f"{setting}{dataset_name}")
    return dataset(root, img_size)