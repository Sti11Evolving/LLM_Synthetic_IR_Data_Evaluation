import ir_datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import re
import time
import openai
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Replace with your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")




# This function will load the irdataset into a dataframe with the query (as text) and documents (as an id) and their associated relevance score (as an int)
# Usually run with dataset "msmarco-passage-v2/trec-dl-2021", but can be run with "msmarco-passage-v2/dev1" for testing
def qrels_dataset(irdataset: ir_datasets.datasets.base.Dataset) -> pd.DataFrame:
    # Do wierd stuff to get the qrels dictionary (Might be a better way of doing this, but it works for now)
    qrels_dict = irdataset.qrels_handler().EXTENSIONS['qrels_dict'](
        irdataset.qrels_handler())

    docs = irdataset.docs_store()

    qrel_dataset = []
    # Iterated over each query and add all associated documents to the list
    for query in irdataset.queries_iter():
        if query.query_id in qrels_dict:
            for doc_id, relevance in qrels_dict[query[0]].items():
                qrel_dataset.append(
                    (query.query_id, doc_id, query.text, docs.get(doc_id).text, relevance, False))

    # Transform to pandas dataframe
    qrel_dataset = pd.DataFrame(qrel_dataset, columns=[
                                "QueryID", "DocumentID", "Query", "Document", "Relevance Actual", "Is Example"])

    # Set 20 of each score as an example
    qrel_dataset.loc[pd.concat([
        qrel_dataset.loc[qrel_dataset['Relevance Actual'] == score].sample(n=20, random_state=3-score) for score in range(4)]).index, ['Is Example']] = True
    
    return qrel_dataset


def load_dataset(file_name: str, dataset_name: str) -> pd.DataFrame:
    if Path.exists(Path(file_name)):
        return pd.read_csv(file_name, index_col=0)
    else:
        dataset = qrels_dataset(ir_datasets.load(dataset_name))
        dataset.to_csv(Path(file_name))
        return dataset


# returns a list of n examples from the example pool
def get_examples(dataset: pd.DataFrame, n: int) -> list[list[str, str, int]]:
    return dataset.loc[dataset['Is Example'] == True, ['Query', 'Document', "Relevance Actual"]].sample(n).values


# A helper function that takes a query and a document and formats it into a prompt
def qd_to_prompt(query: str, document: str) -> str:
    return f"QUERY: {query}\nDOCUMENT: {document}"


# Asks the model to predict the relevance score
def predict_relevance(model: str, query: str, document: str, system_instructions: str, examples: list[list[str, str, int]] = []) -> int:
    # Format all of the pieces together into a single prompt to hand to the model
    messages = [{"role": "system", "content": system_instructions}]

    # Add examples (if any) to the list of messages
    for q, d, s in examples:
        messages.append({"role": "user", "content":  qd_to_prompt(q, d)})
        messages.append({"role": "assistant", "content":  f"RELEVANCY: {s}"})

    # Add query/document to predict
    messages.append(
        {"role": "user", "content":  qd_to_prompt(query, document)})

    # Call the model
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=.2,
        max_tokens=7
    ).choices[0].message.content

    # Assumes that the first number found is the answer
    relevance_grade = int(re.sub('[^0-9]', '', response))

    return relevance_grade


def prediction_metrics(predictions: pd.DataFrame):
    # Get confusion matrix of dataset
    from sklearn.metrics import confusion_matrix, classification_report
    labels = [0, 1, 2, 3]

    # Convert relevancy scores to numpy array
    relavancy_test = predictions.loc[:,
                                     ("Relevance Actual", "Relevance Predicted")].to_numpy()

    # Remove all rows with -1
    relavancy_test = relavancy_test[(relavancy_test != -1).all(axis=1), :]

    actual = relavancy_test[:, 0]
    predicted = relavancy_test[:, 1]

    print(f"Num Samples: {relavancy_test.shape[0]}")
    print(classification_report(actual, predicted, labels=labels))

    cm = confusion_matrix(actual, predicted, labels=labels)

    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)


# Takes the name of a file and returns the path that would be in the current working directory
def file_to_path(file_name: str) -> str:
    return os.getcwd() + '\\' + file_name


def load_experiments(file_name: str, dataset: pd.DataFrame) -> pd.DataFrame:
    experiments_path = file_name
    if Path.exists(Path(experiments_path)):
        # Load from previous tests if they exist
        return pd.read_csv(experiments_path, index_col=0)
    else:
        # Create new experiments file
        experiments = dataset.loc[dataset['Is Example'] == False, ["QueryID", "DocumentID", "Relevance Actual"]].copy()
        experiments["Relevance Predicted"] = -1
        experiments.to_csv(Path(file_name))
        return experiments
    

def run_experiment(system_instruction: str, file_name: str, num_examples: int, dataset: pd.DataFrame, verbose: bool = True) -> None:
    experiments_data = load_experiments(file_name, dataset)

    for (i, (query, document, rel_score, prediction)) in zip(experiments_data.index, experiments_data.values):

        # Skip any qrels that we have already predicted
        while prediction == -1:
            try:
                prediction = predict_relevance(
                    "gpt-3.5-turbo", query, document, system_instruction, get_examples(dataset, num_examples))
                experiments_data.loc[i, "Relevance Predicted"] = prediction
                if verbose:
                    print(f"{i} -- Actual: {rel_score}, Predicted: {prediction}")

                # Save to file
                experiments_data.to_csv(Path(file_name))

            # Handle errors that may occur
            except openai.error.RateLimitError:
                if verbose:
                    print("Model overloaded. Trying again...")
                time.sleep(5)
            except openai.error.APIConnectionError:
                if verbose:
                    print("Failed to connect to OpenAI API. Trying again...")
                time.sleep(5)
            except openai.error.ServiceUnavailableError:
                if verbose:
                    print("Server is down or some other error. Trying again...")
                time.sleep(5)

    if verbose: print("Done!")