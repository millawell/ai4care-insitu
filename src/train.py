import click
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import os
from datetime import date
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd

import configuration as config
from evaluate import report


class BaselineModel:

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

        self.num_offers = int(self.data["Mitarbeiter ID"].apply(len).mean())
        
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


def ungroup(data):
    
    ungroup = []
    for _, row in tqdm(data.iterrows(), desc="ungroup", total=len(data)):
        for ma_id in row["Mitarbeiter ID"]:
            ungroup.append(
                row.to_dict()
            )
            ungroup[-1]["Mitarbeiter ID"] = ma_id

    data = pd.DataFrame(ungroup)

    return data


def run_baseline(train, test):

    model = BaselineModel()
    model.fit(train)

    predictions = []
    for _, row in tqdm(test.iterrows(), desc="prediction", total=len(test)):
        predictions.append(model.predict(row))
    
    print("## BASELINE ## ")
    report(
        test["Mitarbeiter ID"].tolist(),
        predictions
    )


def run_datathon_solution(train, test):

    X_train = train.loc[:, train.columns != 'Mitarbeiter ID']
    y_train = train['Mitarbeiter ID']

    # Transformation Steps
    numeric_features = ['Tag']
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_features = [
        'Einsatzort',
        'PLZ',
        'Qualifikation',
        'Qualifikationgruppe',
        'Schicht',
        'Wochentag',
        'Feiertag'
    ]

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    SVC_pipeline = Pipeline([('preprocessor', preprocessor),
        ('clf', OneVsRestClassifier(SGDClassifier(loss="log"), n_jobs=-1))
    ])

    model = SVC_pipeline.fit(X_train,y_train)


    predictionsprob = model.predict_proba(test)
    mitarbeiter = model.steps[1][1].classes_
    predictions = []
    for index,i in enumerate(predictionsprob):
        mitarbeiter_list = []
        for j in i.argsort()[::-1][:10]: #Get highest probabilities
                        
            if i[j]>=0.0008: #threshold for probability of mitarbeiter.  
                mitarbeiter_list.append(mitarbeiter[j])
        predictions.append(mitarbeiter_list)

    print("## Datathon solution ## ")
    report(
        test["Mitarbeiter ID"].tolist(),
        predictions
    )
    return predictions
    

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

    trainset = pd.read_csv(trainset, sep="\t").sample(frac=1).reset_index(drop=True)
    trainset["Mitarbeiter ID"] = trainset["Mitarbeiter ID"].apply(json.loads)
    trainset.loc[:, "Tag"] = trainset["Tag"].apply(date.fromisoformat)
    min_day = trainset.Tag.min()
    trainset.loc[:, "Tag"] = trainset.Tag.apply(lambda d: (d - min_day).days)

    testset = pd.read_csv(testset, sep="\t")
    testset.loc[:, "Tag"] = testset["Tag"].apply(date.fromisoformat)
    testset.loc[:, "Tag"] = testset.Tag.apply(lambda d: (d - min_day).days)
    testset["Mitarbeiter ID"] = testset["Mitarbeiter ID"].apply(json.loads)
    
    # For Debug 
    # trainset = trainset.sample(1000)
    # testset = testset.sample(1000)

    ungrouped_trainset = ungroup(trainset)

    run_baseline(trainset, testset)

    predictions = run_datathon_solution(ungrouped_trainset, testset)

    testset["Mitarbeiter ID"] = predictions

    testset.to_csv(output, index=False, sep="\t")

    # results on the "small" real world data set with 
    # ~50k orders and ~500k offers in the training set
    # 17k orders in the test set

    # ## BASELINE ##
    # precision at k=1: 0.15575
    # precision at k=2: 0.12922
    # precision at k=3: 0.11385
    # precision at k=5: 0.09608
    # precision at k=10: 0.08229
    # recall at k=1: 0.07707
    # recall at k=2: 0.10906
    # recall at k=3: 0.12964
    # recall at k=5: 0.15971
    # recall at k=10: 0.20232

    # ## Datathon solution ##
    # precision at k=1: 0.28013
    # precision at k=2: 0.25015
    # precision at k=3: 0.22822
    # precision at k=5: 0.20171
    # precision at k=10: 0.17040
    # recall at k=1: 0.05244
    # recall at k=2: 0.08593
    # recall at k=3: 0.11066
    # recall at k=5: 0.14458
    # recall at k=10: 0.20309

if __name__ == '__main__':
    main()