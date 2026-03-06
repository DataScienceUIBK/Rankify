from rankify.utils.pre_defined_datasets import HF_PRE_DEFIND_DATASET
import pandas as pd
from prettytable import PrettyTable

def get_datasets_info(file=None):
    table = PrettyTable(['Retriever', 'Dataset', 'Original ext', 'Compressed','Desc','URL'])
    for retriever, datasets in HF_PRE_DEFIND_DATASET.items():
        for dataset_name, dataset_info in datasets.items():
            
            flattened_entry = {
                'retriever': retriever,
                'dataset': dataset_name,
                'original_ext': dataset_info.get('original_ext'),
                'compressed': dataset_info.get('compressed'),
                'desc': dataset_info.get('desc'),
                'url': dataset_info.get('url')
            }
            table.add_row(flattened_entry.values())
            
    print(table, file=file)