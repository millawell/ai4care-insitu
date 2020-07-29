# Usage
All scripts are in the `src` folder.

## `python synthesize.py -o ../data/`

Creates three files:
* train.tsv (the training data)
* test_publish.tsv (the file that is shared with the public)
* test_truth.tsv (the file that InSitu will use to evaluate the model)

## `python train.py -t ../data/train.tsv -p ../data/test_publish.tsv -o ../data/predictions.tsv`

Train a dummy model and create a file:
* predictions.tsv
Submit your own `train.py` file that has the same signature. (For example by making a copy and just modify the `train` method).

## `python evaluate.py -p ../data/predictions.tsv -t ../data/test_truth.tsv`

Evaluate the predictions from the model (comparing predictions.tsv and test_truth.tsv).


## Data overview

|METADATA||
|:-----|------|
|**topic**||
|**soure**|InSitu|
|**type**|Matching between jobs and health care workers from temp staffing agencies|
|**format**|TSV|
|**licence**|ODbL|
|**publisher**|InSitu|
|**date interval**|2013-08-01 to 2020-12-31|

## Short description
This is a synthetic data set that mimics orders and offers on InSitu.

## Columns

### Einsatzort
`int` 
Identifier of the jobsite (e.g. ward in a hospital or in a nursing home) where the job will take place.
### PLZ
`int`  
Identifier of the postal code of where the job will take place.

### Qualifikation
`int`  
Identifier of the demanded qualification for the job. 

### Qualifikationgruppe
`int`  
Identifier of the group of qualifications that the demanded qualification is part of. The qualifications are grouped based on similarity of the demanded work.

### Tag
`str`  
Day of the job in Isoformat.

### Schicht
`int`  
Identifier of the shift of the job. There are four shifts that define intervals of working time during the day: "Frühdienst", "Zwischendienst", "Spätdienst" and "Nachtdienst".

### Wochentag
`str`  
Day of the week

### Feiertag
`bool`  
Whether the day "Tag" is a holiday in the given postal code.

### Mitarbeiter ID
This is the label that is supposed to predicted during the Datathon.
`JSON list of ints`  
A list of identifiers of workers that have been offered.

