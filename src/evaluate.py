import click
import json
import configuration as config

import numpy as np
import pandas as pd

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

    scores = []
    for (_,pred), (_,true) in zip(predictions.iterrows(), groundtruth.iterrows()):
        scores.append(
            len(set(pred["Mitarbeiter ID"]) & set(true["Mitarbeiter ID"]))
            / len(set(pred["Mitarbeiter ID"]) | set(true["Mitarbeiter ID"]))
        )

    print("accuarcy: {:.5f}".format(
        np.mean(scores)
    ))

    

if __name__ == '__main__':
    main()