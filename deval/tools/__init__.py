from .datasets import (
    GenericDataset,
    WikiDataset,
    AttributionDataset
)
from .selector import Selector

DATASETS = {
    GenericDataset.name: GenericDataset, 
    AttributionDataset.name: AttributionDataset
}
 

  