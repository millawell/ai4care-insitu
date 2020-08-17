import warnings
import click
import json
import configuration as config

import numpy as np
import pandas as pd


def recall_at_k(true, pred, k):

    if any(map(lambda x:len(x)==0, true)):
        warnings.warn(f"Some samples have fewer no true labels.")

    scores = []
    for true_, pred_ in zip(true, pred):
        if len(true_) == 0:
            scores.append(0)
        else:
            scores.append(
                len(set(pred_[:k]) & set(true_))
                / len(true_)
            )

    return np.mean(scores)

def precision_at_k(true, pred, k):

    if any(map(lambda x:len(x)<k, pred)):
        warnings.warn(f"Some samples have fewer than `k={k}` predictions.")

    scores = []
    for true_, pred_ in zip(true, pred):
        if len(pred_) == 0:
            scores.append(0)
        else:
            scores.append(
                len(set(pred_[:k]) & set(true_))
                / len(pred_[:k])
            )

    return np.mean(scores)


def report(true, pred):

    for k in [1,2,3,5,10]:
        print("precision at k={}: {:.5f}".format(
            k,
            precision_at_k(true, pred, k)
        ))
    
    for k in [1,2,3,5,10]:
        print("recall at k={}: {:.5f}".format(
            k,
            recall_at_k(true, pred, k)
        ))
    


@click.command()
@click.option(
    "-p",
    "--predictions",
    required=True,
    type=click.Path(),
    help='Path to prediction set CSV'
)
@click.option(
    "-t",
    "--groundtruth",
    required=True,
    type=click.Path(),
    help='Path to ground truth CSV'
)
def main(predictions, groundtruth):

    predictions = pd.read_csv(predictions, sep="\t")
    predictions["Mitarbeiter ID"] = predictions["Mitarbeiter ID"].apply(json.loads)
    groundtruth = pd.read_csv(groundtruth, sep="\t")
    groundtruth["Mitarbeiter ID"] = groundtruth["Mitarbeiter ID"].apply(json.loads)

    true = groundtruth["Mitarbeiter ID"].tolist()
    pred = predictions["Mitarbeiter ID"].tolist()

    report(true, pred)
if __name__ == '__main__':
    main()