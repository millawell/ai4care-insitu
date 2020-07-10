import click
import json

import pandas as pd

import configuration as config

def train(data):
    '''Dummy trainer that always predicts the same k employee ids'''
    return lambda x: list(range(config.predict_top_k))

@click.command()
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
    help='Path to test set TSV you have to make predictions for'
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(),
    help='Path to where the predictions should be stored'
)
def main(trainset, testset, output):

    trainset = pd.read_csv(trainset, sep="\t")
    trainset["Mitarbeiter ID"] = trainset["Mitarbeiter ID"].apply(json.loads)
    testset = pd.read_csv(testset, sep="\t")

    model = train(trainset)
    testset["Mitarbeiter ID"] = testset.T.apply(model)

    testset.to_csv(output, index=False, sep="\t")

if __name__ == '__main__':
    main()