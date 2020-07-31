import click
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import os
from datetime import date
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
        

        



#def train(data):
#    '''Dummy trainer that always predicts the same k employee ids'''
#
#
#    model = Model()
#    model.fit(data)
#
#    return model

def train(data):
    X_train = data.loc[:, data.columns != 'Mitarbeiter.ID']
    y_train = data['Mitarbeiter.ID']

    # Transformation Steps
    numeric_features = ['Tag']
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_features = ['Einsatzort', 'PLZ', 'Qualifikation', 'Qualifikationgruppe', 'Schicht','Wochentag','Feiertag']
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    SVC_pipeline = Pipeline([('preprocessor', preprocessor),
                ('clf', OneVsRestClassifier(SGDClassifier(loss="log"), n_jobs=1))
            ])
    SVC_pipeline.fit(X_train,y_train)
    return SVC_pipeline


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
    #trainset["Mitarbeiter ID"] = trainset["Mitarbeiter ID"].apply(json.loads)
    trainset.loc[:, "Tag"] = trainset["Tag"].apply(date.fromisoformat)
    min_day = trainset.Tag.min()
    trainset.loc[:, "Tag"] = trainset.Tag.apply(lambda d: (d - min_day).days)
    
    testset = pd.read_csv(testset, sep="\t")

    testset.loc[:, "Tag"] = testset["Tag"].apply(date.fromisoformat)
    testset.loc[:, "Tag"] = testset.Tag.apply(lambda d: (d - min_day).days)
	trainset.drop(['Unnamed: 0','X.1','X'],axis=1,inplace=True) # remove weird columns from preprocessing of training data 
    

    model = train(trainset)
    predictionsprob = model.predict_proba(testset)
    mitarbeiter = model.steps[1][1].classes_
    solutions = {}
    for index,i in enumerate(predictionsprob):
        mitarbeiter_list = []
        for j in i.argsort()[-8:]: #Get 8 highest probabilities
                        
            if i[j]>=0.0008: #threshold for probability of mitarbeiter.  
                mitarbeiter_list.append(mitarbeiter[j])
        solutions[index] = mitarbeiter_list
    
    pd.Series(solutions).rename("Mitarbeiter ID").to_csv(os.path.join('data','predictions.tsv'),sep="\t")
    
#    preds = []
#    for _, row in tqdm(testset.iterrows(), desc="prediction", total=len(testset)):
#        preds.append(model.predict(row))
#    testset["Mitarbeiter ID"] = preds

#    testset.to_csv(output, index=False, sep="\t")

if __name__ == '__main__':
    main()