import os
import pickle
import re
import time
from pathlib import Path

import ir_datasets
import matplotlib.pyplot as plt
import openai
import pandas as pd
import seaborn as sn
from dotenv import load_dotenv, set_key
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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


# parses the response and returns a number
def number_parser(content: str) -> int:
    try:
        words = content.split()
        return int(re.sub(r'[.]', '', words[words.index('RELEVANCY:')+1]))
    except Exception as e:
        raise ValueError(f'Malformed response: {content}')
    

# parses the response and returns a word
def word_parser(content: str) -> str:
    try:
        words = content.split(' ')
        return words[words.index('RELEVANCY:')+1]
    except Exception as e:
        raise ValueError(f'Malformed response: {content}')


# parses for a yes or no response
def yn_parser(content: str) -> str:
    for no_word in ['no', 'not relevent', 'not related', 'unrelated']:
        if no_word in content.lower():
            return 'False'

    for yes_word in ['yes', 'relevant', 'related']:
        if yes_word in content.lower():
            return 'True'

    raise ValueError(f'Malformed response: {content}')


# Asks the model to predict the relevance score
def predict_relevance(model: str, query: str, document: str, system_instructions: str, parser: callable, examples: list[list[str, str, int]] = [], max_tokens: int = 7, temperature: int = 0) -> tuple[int | str, int, int]:
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
        temperature=temperature,
        max_tokens=max_tokens
    )
    content = response['choices'][0]['message']['content']
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']

    finish_reason = response['choices'][0]['finish_reason']
    if (finish_reason == 'length'):
        print(f'Model hit maximum token limit ({max_tokens}). Consider increasing token limit.\nMessage generated: {content}')


    # Attempt to parse the content for the models prediction
    try:
        relevance_grade = parser(content)
    except Exception as e:
        raise e

    return relevance_grade, prompt_tokens, completion_tokens



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
        experiments = dataset.loc[dataset['Is Example'] == False, ["Query", "Document", "Relevance Actual"]].copy()
        experiments["Relevance Predicted"] = None
        experiments.to_csv(Path(file_name))
        return experiments
    

def run_experiment(experiment_name: str, model: str, system_instruction: str, num_examples: int, dataset: pd.DataFrame, parser: callable = number_parser, max_tokens: int = 7, temperature: int = 0, max_generated: float = float('inf'), verbose: bool = True) -> None:
    experiment_path = f'Data/{experiment_name}.csv'
    experiments_data = load_experiments(experiment_path, dataset)
    to_run = experiments_data.loc[pd.isna(experiments_data['Relevance Predicted'])]

    for (i, (query, document, rel_score, prediction)) in zip(to_run.index, to_run.values):
        if i > max_generated: break

        # Skip any qrels that we have already predicted
        while True:
            try:
                prediction, prompt_tokens, completion_tokens = predict_relevance(
                    model, query, document, system_instruction, parser, get_examples(dataset, num_examples), max_tokens, temperature)
                experiments_data.loc[i, 'Relevance Predicted'] = prediction
                if verbose:
                    print(f"{i} -- Actual: {rel_score}, Predicted: {prediction}")

                # Save to file
                experiments_data.to_csv(Path(experiment_path))
                increment_tokens_and_save(prompt_tokens, completion_tokens, experiment_name)
                break

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


# Function generated by chatGPT
def increment_tokens_and_save(prompt_tokens: int, completion_tokens: int, experiment: str) -> None:
    """
    Increments the 'prompt_tokens' and 'completion_tokens' values stored in a .env file for a specific experiment and saves the updated values back to the file.

    Parameters:
        prompt_tokens (int): The integer value to increment the 'prompt_tokens' by for the specified experiment.
        completion_tokens (int): The integer value to increment the 'completion_tokens' by for the specified experiment.
        experiment (str): The name of the experiment for which to update the token counts.

    Returns:
        None: This function does not return anything. The updated values are saved to the .env file directly.

    Raises:
        ValueError: If the provided prompt_tokens or completion_tokens are not positive integers.
    """
    # Step 1: Load existing .env file or create a new one if it doesn't exist
    if os.path.isfile('.env'):
        load_dotenv(override=True)
    else:
        with open('.env', 'w') as file:
            pass

    # Step 2: Validate the provided arguments
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        raise ValueError("Invalid argument types. Both prompt_tokens and completion_tokens must be integers.")

    # Step 3: Validate that the provided integers for input and output tokens are positive
    if prompt_tokens <= 0 or completion_tokens <= 0:
        raise ValueError("The prompt_tokens and completion_tokens must be positive integers.")

    # Step 4: Increment the 'prompt_tokens' and 'completion_tokens' values with the provided integers
    prompt_tokens_key = f'{experiment}_prompt_tokens'
    completion_tokens_key = f'{experiment}_completion_tokens'
    current_prompt_tokens = int(os.getenv(prompt_tokens_key, 0))
    current_completion_tokens = int(os.getenv(completion_tokens_key, 0))
    new_prompt_tokens = current_prompt_tokens + prompt_tokens
    new_completion_tokens = current_completion_tokens + completion_tokens

    # Step 5: Save the updated values to the .env file for the specific experiment
    set_key('.env', prompt_tokens_key, str(new_prompt_tokens))
    set_key('.env', completion_tokens_key, str(new_completion_tokens))



# Function generated by ChatGPT
def get_tokens_for_experiment(experiment: str) -> tuple:
    """
    Retrieves the number of input and output tokens for a specific experiment from the .env file.

    Parameters:
        experiment (str): The name of the experiment for which to retrieve the token counts.

    Returns:
        tuple: A tuple containing the number of input tokens and output tokens for the specified experiment.

    Raises:
        FileNotFoundError: If the .env file does not exist.
        ValueError: If the provided experiment name does not exist in the .env file.
    """
    # Step 1: Check if the provided experiment name exists in the .env file
    if not os.path.isfile('.env'):
        raise FileNotFoundError("The .env file does not exist.")

    # Step 2: Load the .env file
    load_dotenv(override=True)

    # Step 3: Get the values of 'prompt_tokens' and 'completion_tokens' for the specific experiment
    prompt_tokens_key = f'{experiment}_prompt_tokens'
    completion_tokens_key = f'{experiment}_completion_tokens'

    if prompt_tokens_key not in os.environ or completion_tokens_key not in os.environ:
        raise ValueError(f"The experiment '{experiment}' does not exist in the .env file.")

    prompt_tokens = int(os.environ[prompt_tokens_key])
    completion_tokens = int(os.environ[completion_tokens_key])

    # Step 4: Return the token counts as a tuple
    return prompt_tokens, completion_tokens


# Function generated by ChatGPT
def reset_tokens(experiment: str) -> None:
    """
    Resets the 'prompt_tokens' and 'completion_tokens' values for a specific experiment in the .env file to 0.

    Parameters:
        experiment (str): The name of the experiment for which to reset the token counts.

    Returns:
        None: This function does not return anything. The values are reset to 0 in the .env file directly.

    Raises:
        FileNotFoundError: If the .env file does not exist.
    """
    # Step 1: Check if the .env file exists
    if not os.path.isfile('.env'):
        raise FileNotFoundError("The .env file does not exist.")

    # Step 2: Load the .env file
    load_dotenv(override=True)

    # Step 3: Reset the 'prompt_tokens' and 'completion_tokens' values to 0 for the specific experiment
    prompt_tokens_key = f'{experiment}_prompt_tokens'
    completion_tokens_key = f'{experiment}_completion_tokens'

    set_key('.env', prompt_tokens_key, '0')
    set_key('.env', completion_tokens_key, '0')

    print(f"Tokens reset for experiment '{experiment}'. The .env file updated successfully.")