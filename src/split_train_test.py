import click
import json
from bisect import bisect_left
from tqdm import tqdm
from copy import deepcopy as dcp
import numpy as np
from datetime import datetime

import pandas as pd

import configuration as config
from sklearn.model_selection import train_test_split

def is_in(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return True
    return False


@click.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(),
    help='Path to where the whole dataset is stored'
)
@click.option(
    "-t",
    "--trainset",
    required=True,
    type=click.Path(),
    help='Path to train set TSV'
)
@click.option(
    "-p",
    "--testset",
    required=True,
    type=click.Path(),
    help='Path to test set TSV'
)
def main(input, trainset, testset):

    whole = pd.read_csv(input, sep="\t")
    whole["Mitarbeiter ID"] = whole["Mitarbeiter ID"].apply(json.loads)
    whole.loc[:, "Tag"] = whole["Tag"].apply(datetime.fromisoformat)

    # whole = pd.DataFrame(whole[np.logical_and(
    #     whole.Tag >= datetime(2019,1,1),
    #     whole.Tag < datetime(2019,4,1)
    # )])

    train, test = train_test_split(whole, random_state=341223)

    all_train_ma_ids = sorted(list(set([it for ll in train["Mitarbeiter ID"] for it in ll])))

    should_not_belong_to_test = []
    for irow, row in tqdm(
        test.iterrows(),
        total=len(test),
        desc="check test set for rows that don't occur in training set"
    ):
        for ma_id in row["Mitarbeiter ID"]:
            if not is_in(all_train_ma_ids, ma_id):
                should_not_belong_to_test.append(irow)

    train = pd.concat([
        train,
        dcp(test.loc[should_not_belong_to_test])
    ])
    
    test = dcp(test.loc[~test.index.isin(should_not_belong_to_test)])

    train.to_csv(trainset, index=False, sep="\t")
    test.to_csv(testset, index=False, sep="\t")

if __name__ == '__main__':
    main()