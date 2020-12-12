import pandas as pd

def tsv_max_entry_value(path, entry) ->int:
    data = pd.read_csv(path, delimiter="\t")[entry]
    return int(max(data))