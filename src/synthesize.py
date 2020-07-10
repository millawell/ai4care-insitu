import os
from datetime import datetime, timedelta
import click
from copy import deepcopy as dcpy

import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from sklearn.model_selection import train_test_split
import pandas as pd 
import holidays

import configuration as config

@click.command()
@click.option(
    "-o",
    "--outdir",
    required=True,
    type=click.Path(),
    help='Path to directory where the synthetic datasets should be stored'
)
def main(outdir):
    rng = RandomState(MT19937(SeedSequence(config.seed)))

    berlin_holidays = holidays.DE(prov="BW")

    num_employees = 20000
    num_jobsites = 200
    num_areas = 20
    num_qualifications = 40
    num_shifts = 3
    num_days = 356

    num_orders = 1000
    df = pd.DataFrame.from_dict({
        "Einsatzort": rng.randint(0, num_jobsites, num_orders),
        "Qualifikation":rng.randint(0, num_qualifications, num_orders),
        "Schicht": rng.randint(0, num_shifts, num_orders),
        "Tag": rng.randint(0, num_days, num_orders),
    })

    df["Tag"] = df["Tag"].apply(lambda day: datetime(2019, 1, 1)+ timedelta(day))
    df["Wochentag"] = df["Tag"].apply(lambda day: day.strftime("%a"))
    df["Feiertag"] = df["Tag"].apply(lambda day: day in berlin_holidays)

    # grouping of jobsites into areas
    area_splits = np.cumsum(rng.randint(1,10,num_areas))
    area_splits = (area_splits.T / area_splits.max()*num_jobsites).astype(int)
    df["Ort"] = df["Einsatzort"].apply(lambda jobsite_id: np.argmax(area_splits>jobsite_id))

    offers = []
    for _ in range(len(df)):
        offers.append(
            rng.choice(range(num_employees), replace=False, size=rng.randint(1,6)).tolist()
        )

    df["Mitarbeiter ID"] = offers


    train, test = train_test_split(df)
    
    train.to_csv(
        os.path.join(outdir, "train.tsv"),
        index=False,
        sep="\t"
    )
    test.to_csv(
        os.path.join(outdir, "test_truth.tsv"),
        index=False,
        sep="\t"
    )
    test[["Einsatzort", "Qualifikation", "Schicht", "Tag", "Wochentag", "Feiertag", "Ort"]].to_csv(
        os.path.join(outdir, "test_publish.tsv"),
        index=False,
        sep="\t"
    )


if __name__ == '__main__':
    main()