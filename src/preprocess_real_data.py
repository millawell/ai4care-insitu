import click
import configuration as config

from datetime import datetime
import holidays

import pandas as pd

def parse_prov(in_):
    if isinstance(in_, str) and "-" in in_:
        return holidays.DE(prov=in_.split("-")[1])
    return holidays.DE()
        

def make_categorical(df, label_col):
    lookup = sorted(df[label_col].unique().tolist())
    return df[label_col].apply(lambda x: lookup.index(x))


@click.command()
@click.option(
    "-i",
    "--inputdata",
    required=True,
    type=click.Path(),
    help='Path to raw CSV'
)
@click.option(
    "-o",
    "--outputdata",
    required=True,
    type=click.Path(),
    help='Path to where the preprocessed TSV should be outputted'
)
def main(inputdata, outputdata):

    df = pd.read_csv(
        inputdata,
        sep=",",
        index_col=False,
        header=None,
        low_memory=False,
        names=
        [
            "Mitarbeiter ID",
            "Einsatzort",
            "Ort",
            "stadt",
            "PLZ",
            "anfrage_qualifikation",
            "Qualifikation",
            "Qualifikationgruppe",
            "angebots_qualifikation",
            "angebots_qualifikation_label",
            "angebots_qualifikation_gruppe",
            "Schicht",
            "Wochentag",
            "Tag",
            "status",
        ]
    )

    df = df[[
        "Mitarbeiter ID",
        "Einsatzort",
        "Ort",
        "PLZ",
        "Qualifikation",
        "Qualifikationgruppe",
        "Schicht",
        "Wochentag",
        "Tag"
    ]]

    df["Mitarbeiter ID"] = make_categorical(df, "Mitarbeiter ID")

    df["Bundesland"] = df["Ort"].apply(lambda x: x if isinstance(x, str) else "DE")
    df.loc[ df["PLZ"].isnull(), "PLZ"] = 0.0
    
    grouped_df = []
    grouper = df.groupby(["Einsatzort", "PLZ", "Bundesland", "Qualifikation", "Qualifikationgruppe", "Schicht", "Wochentag", "Tag"])
    for (jobsite, plz, area, quali, quali_group, shift, weekday, day), subdf in grouper:
        grouped_df.append({
            "Einsatzort": jobsite,
            "PLZ": plz, 
            "Bundesland": area,
            "Qualifikation": quali,
            "Qualifikationgruppe": quali_group,
            "Schicht": shift,
            "Wochentag": weekday,
            "Tag": day,
            "Mitarbeiter ID": subdf["Mitarbeiter ID"].tolist()
        })

    df = pd.DataFrame(grouped_df)

    df["province_holidays"] = df["Bundesland"].apply(parse_prov)

    df["Tag"] = df["Tag"].apply(datetime.fromisoformat)
    df["Wochentag"] = df["Tag"].apply(lambda day: day.strftime("%a"))
    df["Feiertag"] = df.T.apply(lambda row: None).T
    df["Feiertag"] = df.T.apply(lambda row: row["Tag"] in row["province_holidays"]).T

    df["PLZ_label"] = df["PLZ"]
    df["PLZ"] = make_categorical(df, "PLZ")

    df["Qualifikation_label"] = df["Qualifikation"]
    df["Qualifikation"] = make_categorical(df, "Qualifikation")
    df["Qualifikationgruppe"] = make_categorical(df, "Qualifikationgruppe")

    df["Schicht_label"] = df["Schicht"]
    df["Schicht"] = make_categorical(df, "Schicht")

    df["Wochentag_label"] = df["Wochentag"]
    df["Wochentag"] = make_categorical(df, "Wochentag")

    df[["Einsatzort",
        "PLZ",
        "PLZ_label",
        "Qualifikation",
        "Qualifikation_label",
        "Qualifikationgruppe",
        "Schicht",
        "Schicht_label",
        "Tag",
        "Wochentag",
        "Wochentag_label",
        "Feiertag",
        "Mitarbeiter ID"
    ]].to_csv(
        outputdata,
        index=False,
        sep="\t"
    )

if __name__ == '__main__':
    main()