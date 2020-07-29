import click
import json
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd

import configuration as config


class Model:

    def fit(self, data):

        self.data = data[[
            'Einsatzort',
            'PLZ',
            'Qualifikation',
            'Qualifikationgruppe',
            'Tag',
            'Mitarbeiter ID'
        ]]

        self.data = self.data.set_index([
            "Einsatzort",
            "PLZ",
            "Qualifikation",
            "Qualifikationgruppe"
        ])

        self.num_offers = 4 ##int(self.data["Mitarbeiter ID"].apply(len).mean())
        
        return self

    def predict(self, X):
        """Find offers in the training set that match the features in the test sample and 
           return that employee ids that are temporally closest to the test sample date. 
           The matching ist two-fold. A fine-grained matching (same jobsite and same qualification)
           and a coarse-grained matching (same postal code and same qualification group). The latter
           will only be used if the former does not return any training samples."""
        try:
            subset = pd.DataFrame(self.data.loc[(
                X.Einsatzort,
                slice(None),
                X.Qualifikation,
                slice(None)
            ),:])
        except KeyError:
            try:
                subset = pd.DataFrame(self.data.loc[(
                    slice(None),
                    X.PLZ,
                    slice(None),
                    X.Qualifikationgruppe
                ),:])
            except KeyError:
                return []
        
        subset["days_dist"] = np.abs(subset.Tag - X.Tag)
        y_pred = [
            it for ll in subset.sort_values("days_dist")[::-1]["Mitarbeiter ID"] 
                for it in ll]
        return y_pred[:self.num_offers]
        

        



def train(data):
    '''Dummy trainer that always predicts the same k employee ids'''
    

    model = Model()
    model.fit(data)

    return model

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
    trainset.loc[:, "Tag"] = trainset["Tag"].apply(datetime.fromisoformat)
    min_day = trainset.Tag.min()
    trainset.loc[:, "Tag"] = trainset.Tag.apply(lambda d: (d - min_day).days)
    
    testset = pd.read_csv(testset, sep="\t")

    testset.loc[:, "Tag"] = testset["Tag"].apply(datetime.fromisoformat)
    testset.loc[:, "Tag"] = testset.Tag.apply(lambda d: (d - min_day).days)


    model = train(trainset)

    preds = []
    for _, row in tqdm(testset.iterrows(), desc="prediction", total=len(testset)):
        preds.append(model.predict(row))
    testset["Mitarbeiter ID"] = preds

    testset.to_csv(output, index=False, sep="\t")

if __name__ == '__main__':
    main()