import os
from datetime import datetime, timedelta
import click
from copy import deepcopy as dcpy

from tqdm import tqdm

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

    num_employees = 50000

    num_orders = 1000000

    num_jobsites = 2800
    num_areas = 180
    num_qualifications = 214
    num_qualigroups = 13
    num_shifts = 4
    num_days = 2708

    start_day = datetime(2013, 8, 1)

    print("create sliding window of active employees")
    active_employees = np.zeros((num_employees, num_days)).astype(bool)

    left = 0
    right = 100
    upkeep = 400
    change = (.95,1-.95)
    for irow, row in enumerate(active_employees):
        active_employees[irow,left:right] = 1
        left = left + rng.choice([0,1], p=change)
        right = left + upkeep + rng.choice([0,1], p=change)


    print("create base distributions for areas, qualis and shifts")
    areas = rng.dirichlet(np.ones(num_areas)*.1)

    jobsites = rng.dirichlet(np.ones(num_jobsites)*.1)

    area_of_jobsite = np.empty(num_jobsites)
    for ijobsite, jobsite in enumerate(jobsites):
        area_of_jobsite[ijobsite] = rng.choice(np.arange(num_areas), p=areas)
        
    qualigroups = rng.dirichlet(np.ones(num_qualigroups)*.1)

    qualis = rng.dirichlet(np.ones(num_qualifications)*.1)

    qualigroup_of_quali = np.empty(num_qualifications)
    for iquali, quali in enumerate(qualis):
        qualigroup_of_quali[iquali] = rng.choice(np.arange(num_qualigroups), p=qualigroups)
        
    shifts = rng.dirichlet(np.ones(num_shifts))

    orders = []
    for _ in tqdm(range(num_orders), desc="create orders"):
        shift = rng.choice(range(num_shifts), p=shifts)
        
        jobsite = rng.choice(range(num_jobsites), p=jobsites)
        area = area_of_jobsite[jobsite]
        
        quali = rng.choice(range(num_qualifications), p=qualis)
        qualigroup = qualigroup_of_quali[quali]
        
        day = rng.randint(0,num_days)
        
        orders.append({
            "Schicht": shift,
            "Einsatzort": jobsite,
            "PLZ": area,
            "Qualifikation": quali,
            "Qualifikationgruppe": qualigroup,
            "Tag": day,
        })

    employee_qualifications = rng.multinomial(1, qualis, size=(num_employees)).astype(bool)
    employee_jobsites = rng.multinomial(1, jobsites, size=(num_employees)).astype(bool)

    orders = pd.DataFrame(orders)
    offers = []

    ps = np.ones(6)/np.arange(1,7)
    ps /= ps.sum()

    for _, order in tqdm(orders.iterrows(), desc="create offers", total=len(orders)):
        
        match_active = active_employees[:,int(order.Tag)]
        match_quali = employee_qualifications[:,int(order.Qualifikation)]
        match_jobsite = employee_jobsites[:,int(order.Einsatzort)]

        match, = (match_active & match_quali & match_jobsite).nonzero()

        offers.append(
            match[:6]
        )
        if len(offers[-1]) == 0:

            offers[-1] = rng.choice(
                match_active.nonzero()[0],
                np.random.choice(
                    range(1,7),
                    p=ps
            ))

    berlin_holidays = holidays.DE(prov="BE")

    orders["Mitarbeiter ID"] = offers
    print("add day meta data")
    orders["Tag"] = orders["Tag"].apply(lambda day: start_day+ timedelta(day))
    orders["Wochentag"] = orders["Tag"].apply(lambda day: day.strftime("%a"))
    orders["Feiertag"] = orders["Tag"].apply(lambda day: day in berlin_holidays)

    orders = orders[[
        "Einsatzort",
        "PLZ",
        "Qualifikation",
        "Qualifikationgruppe",
        "Schicht",
        "Tag",
        "Wochentag",
        "Feiertag",
        "Mitarbeiter ID"
    ]]
    orders = orders.sort_values("Tag")


    train, test = train_test_split(orders)
    
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
    test[[
        "Einsatzort",
        "PLZ",
        "Qualifikation",
        "Qualifikationgruppe",
        "Schicht",
        "Tag",
        "Wochentag",
        "Feiertag"
    ]].to_csv(
        os.path.join(outdir, "test_publish.tsv"),
        index=False,
        sep="\t"
    )


if __name__ == '__main__':
    main()