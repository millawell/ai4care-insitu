# Usage
All scripts are in the `src` folder.

## `python synthesize.py -o ../data/`

Creates three files:
* train.tsv (the training data)
* test_publish.tsv (the file that is shared with the public)
* test_truth.tsv (the file that InSitu will use to evaluate the model)

## `python train.py -t ../data/train.tsv -p ../data/test_publish.tsv -o ../data/predictions.tsv`

Train a dummy model and creates a file:
* predictions.tsv
Submit your own `train.py` file that has the same signature. (For example by making a copy and just modify the `train` method).

## `touch evaluate.py -p ../data/predictions.tsv -t ../data/test_truth.tsv`

Evaluate the predictions from the model (comparing predictions.tsv and test_truth.tsv).
