# This repository has all of the code and data used in the paper found here {Update with link to paper when published}

All synthetic data evaluations can be found in the `data/`.

All code was run using python 3.10. To run the code for yourself, run `pip install -r requirements.txt` and ensure that you have the java jdk installed.

Synthetic data is generated using the GPT-3.5-turbo using the openai API based off of the MSMARCO passage v2 retrieved from ir_datasets. You can find all of the functions relating to that in `qrel_reconstruction.py` and use `Reconstruction.ipynb` to run them.

Search and evaluation is done in `IR_testing.ipynb` using pyserini (anserini) libraries for the search models and ir_measures.
