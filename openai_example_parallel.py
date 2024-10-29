"""
API REQUEST PARALLEL PROCESSOR

Using the OpenAI API to process lots of text quickly takes some care.
If you trickle in a million API requests one by one, they'll take days to complete.
If you flood a million API requests in parallel, they'll exceed the rate limits and fail with errors.
To maximize throughput, parallel requests need to be throttled to stay under rate limits.

This script parallelizes requests to the OpenAI API while throttling to stay under rate limits.

Features:
- Streams requests from file, to avoid running out of memory for giant jobs
- Makes requests concurrently, to maximize throughput
- Throttles request and token usage, to stay under rate limits
- Retries failed requests up to {max_attempts} times, to avoid missing data
- Logs errors, to diagnose problems with requests

Example command to call script:
```
python examples/api_request_parallel_processor.py \
  --requests_filepath examples/data/example_requests_to_parallel_process.jsonl \
  --save_filepath examples/data/example_requests_to_parallel_process_results.jsonl \
  --request_url https://api.openai.com/v1/embeddings \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 6250000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20
```

Inputs:
- requests_filepath : str
    - path to the file containing the requests to be processed
    - file should be a jsonl file, where each line is a json object with API parameters
    - e.g., {"model": "text-embedding-ada-002", "input": "embed me"}
    - as with all jsonl files, take care that newlines in the content are properly escaped (json.dumps does this automatically)
    - an example file is provided at examples/data/example_requests_to_parallel_process.jsonl
    - the code to generate the example file is appended to the bottom of this script
- save_filepath : str, optional
    - path to the file where the results will be saved
    - file will be a jsonl file, where each line is an array with the original request plus the API response
    - e.g., [{"model": "text-embedding-ada-002", "input": "embed me"}, {...}]
    - if omitted, results will be saved to {requests_filename}_results.jsonl
- request_url : str, optional
    - URL of the API endpoint to call
    - if omitted, will default to "https://api.openai.com/v1/embeddings"
- api_key : str, optional
    - API key to use
    - if omitted, the script will attempt to read it from an environment variable {os.getenv("OPENAI_API_KEY")}
- max_requests_per_minute : float, optional
    - target number of requests to make per minute (will make less if limited by tokens)
    - leave headroom by setting this to 50% or 75% of your limit
    - if requests are limiting you, try batching multiple embeddings or completions into one request
    - if omitted, will default to 1,500
- max_tokens_per_minute : float, optional
    - target number of tokens to use per minute (will use less if limited by requests)
    - leave headroom by setting this to 50% or 75% of your limit
    - if omitted, will default to 125,000
- token_encoding_name : str, optional
    - name of the token encoding used, as defined in the `tiktoken` package
    - if omitted, will default to "cl100k_base" (used by `text-embedding-ada-002`)
- max_attempts : int, optional
    - number of times to retry a failed request before giving up
    - if omitted, will default to 5
- logging_level : int, optional
    - level of logging to use; higher numbers will log fewer messages
    - 40 = ERROR; will log only when requests fail after all retries
    - 30 = WARNING; will log when requests his rate limits or other errors
    - 20 = INFO; will log when requests start and the status at finish
    - 10 = DEBUG; will log various things as the loop runs to see when they occur
    - if omitted, will default to 20 (INFO).

The script is structured as follows:
    - Imports
    - Define main()
        - Initialize things
        - In main loop:
            - Get next request if one is not already waiting for capacity
            - Update available token & request capacity
            - If enough capacity available, call API
            - The loop pauses if a rate limit error is hit
            - The loop breaks when no tasks remain
    - Define dataclasses
        - StatusTracker (stores script metadata counters; only one instance is created)
        - APIRequest (stores API inputs, outputs, metadata; one method to call API)
    - Define functions
        - api_endpoint_from_url (extracts API endpoint from request URL)
        - append_to_jsonl (writes to results file)
        - num_tokens_consumed_from_request (bigger function to infer token usage from request)
        - task_id_generator_function (yields 1, 2, 3, ...)
    - Run main()
"""

# imports
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass  # for storing API inputs, outputs, and metadata
import pandas as pd

async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")

        while True:
            # get next request (if one is not already waiting for capacity)
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                    logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
                elif file_not_finished:
                    try:
                        # get new request
                        request_json = json.loads(next(requests))
                        next_request = APIRequest(
                            task_id=next(task_id_generator),
                            request_json=request_json,
                            token_consumption=num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name),
                            attempts_left=max_attempts,
                        )
                        status_tracker.num_tasks_started += 1
                        status_tracker.num_tasks_in_progress += 1
                        logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                    except StopIteration:
                        # if file runs out, set flag to stop reading it
                        logging.debug("Read file exhausted")
                        file_not_finished = False

            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity = min(
                available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
                max_requests_per_minute,
            )
            available_token_capacity = min(
                available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
                max_tokens_per_minute,
            )
            last_update_time = current_time

            # if enough capacity available, call API
            if next_request:
                next_request_tokens = next_request.token_consumption
                if (
                    available_request_capacity >= 1
                    and available_token_capacity >= next_request_tokens
                ):
                    # update counters
                    available_request_capacity -= 1
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens
                    next_request.attempts_left -= 1

                    # call API
                    asyncio.create_task(
                        next_request.call_api(
                            request_url=request_url,
                            request_header=request_header,
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker,
                        )
                    )
                    next_request = None  # reset next_request to empty

            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress == 0:
                break

            # main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(seconds_to_sleep_each_loop)

            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
            if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
                await asyncio.sleep(remaining_seconds_to_pause)
                # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

        # after finishing, log final status
        logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
        if status_tracker.num_tasks_failed > 0:
            logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}.")
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    result = []

    async def call_api(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        example_id = self.request_json['example_id']
        copied_request = dict(self.request_json)
        del copied_request['example_id']
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=copied_request
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {copied_request} failed after all attempts. Saving errors: {self.result}")
                append_to_jsonl([example_id, copied_request, [str(e) for e in self.result]], save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            append_to_jsonl([example_id, copied_request, response], save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


# functions


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search('^https://[^/]+/v\\d+/(.+)$', request_url)
    return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

# run script

if __name__ == "__main__":

    bsDetector = True
    if bsDetector:

        # parse command line arguments
        # or set the defaults here and forget about them
        parser = argparse.ArgumentParser()
        # parser.add_argument("--requests_filepath")
        # parser.add_argument("--save_filepath", default=None)
        parser.add_argument("--request_url", default="https://api.openai.com/v1/chat/completions")
        parser.add_argument("--api_key",
                            default=os.getenv("OPENAI_API_KEY")))  # default=os.getenv("OPENAI_API_KEY"))
        parser.add_argument("--max_requests_per_minute", type=int, default=10000 * 0.9)
        parser.add_argument("--max_tokens_per_minute", type=int, default=300000 * 0.9)
        parser.add_argument("--token_encoding_name", default="cl100k_base")
        parser.add_argument("--max_attempts", type=int, default=30)
        parser.add_argument("--logging_level", default=logging.INFO)
        args = parser.parse_args()

        gpt_model = "gpt-4-0314"
        temperature = 1.0

        experiment_name = f'LIAR_new_basic_retrieval'
        base_path = './'

        input_df = pd.read_json(f'resolution_df.jsonl', lines=True)

        input_df['example_id'] = input_df['example_id']
        #input_df['example_id'] = input_df.index  # index usually easiest, but in theory can be anything unique, e.g. the input text itself if that won't be duplicated

        # this bit is for restarting if stopped partway through, e.g. cluster job timeout
        save_filepath = f'{experiment_name}_{gpt_model}_temp{temperature}_rawoutput.jsonl'
        if os.path.exists(save_filepath):
            reload_df = pd.read_json(save_filepath, lines=True, orient='records')
            input_df = input_df[~input_df.example_id.isin(reload_df[0])]


        # print('input_df after reload', input_df.shape)

        def convert_bool(answer):
            try:
                tmp = int(answer)
                if tmp == 1:
                    return 'True'
                else:
                    return 'False'
            except:
                return 'False'

        # this function to set up the prompt
        def text_to_messages(row):

            #default
            text = row['text']
            analysis = ''

            #print(f"this is gpt's answer {row['gpt4-answer']} and this is text: {row['text']}")

            #categorical
            #text = convert_bool(row['text'])
            #analysis = row['gpt-answer']
            #text = re.search(r'"([^"]*)"', row[0]['messages'][0]['content']).group(0)
            #analysis = row[1]['choices'][0]['message']['content']

            #text + analysis
            #text = row['question']
            #analysis = row['gpt-answer']

            # score_then_explain

            paper_prompt = f'Rate the truthfulness of the following statement: "{text}" \
                 Your task is to provide a score from 0 to 100, where 0 represents definitively false and 100 represents definitively true, but you must not state your score until you have presented a thorough analysis. Do not begin your response with a number. First write your analysis, then write a vertical bar "|", then finally state your score.'

            two_step_prompt = f"Statement: '{text}'  \
                              Proposed answer: '{analysis}'  \
            Your task is to rate the uncertainty of the proposed answer on a score from 0 to 100, where 0 represents definitely uncertain and 100 represents definitely certain. Please, only answer with your score. "

            single_step_prompt = f'Rate the truthfulness of the following statement: "{text}”. \
                            Your task is to provide a truthfulness score from 0 to 100, where 0 represents definitively false and 100 represents definitively true, but you must not state your score until you have presented a thorough analysis. First, write your analysis. Then, you are required to state your truthfulness score in squared brackets “[]”. Finally, provide a different uncertainty score, enclosed by vertical lines “|”, where 0 represents definitely uncertain and 100 represents definitely certain of your answer'

            strategy_qa_prompt = f"Question: '{text}'  \
                              Proposed answer: '{analysis}'  \
            Your task is to rate the uncertainty of the proposed answer on a score from 0 to 100, where 0 represents definitely uncertain and 100 represents definitely certain. Please, only answer with your score. "

            context = row['additional-context']
            retrieval_resolution_prompt = f'Rate the truthfulness of the following statement: "{text}" provided this additional context: “{context}”. \
                    Your task is to provide a score from 0 to 100, where 0 represents definitively false and 100 represents definitively true, but you must not state your score until you\'ve presented a thorough analysis. Do not begin your response with a number. First write your analysis, then finally state your score enclosed by squared brackets “[]”.'

            return [{"role": "user", "content": retrieval_resolution_prompt}]




        input_df['messages'] = input_df.apply(lambda row: text_to_messages(row), axis=1)
        input_df['temperature'] = temperature
        input_df['model'] = gpt_model
        input_df['max_tokens'] = 1000  # maximum tokens in the response

        requests_filepath = f'{experiment_name}_{gpt_model}_temp{temperature}_preprocessedforparallel.jsonl'
        input_df[['example_id', 'messages', 'temperature', 'model', 'max_tokens']].to_json(requests_filepath,
                                                                                           lines=True,
                                                                                           orient='records')  # only fields gpt is expecting + example_id
        input_df.to_json(requests_filepath.split('.jsonl')[0] + 'full.jsonl', lines=True,
                         orient='records')  # not really needed, just record of the full input df

        # run script
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=requests_filepath,
                save_filepath=save_filepath,
                request_url=args.request_url,
                api_key=args.api_key,
                max_requests_per_minute=float(args.max_requests_per_minute),
                max_tokens_per_minute=float(args.max_tokens_per_minute),
                token_encoding_name=args.token_encoding_name,
                max_attempts=int(args.max_attempts),
                logging_level=int(args.logging_level),
            )
        )

        # match things back together with example_id, and also get rid of unneeded metadata in the gpt response
        output_df = pd.read_json(save_filepath, lines=True, orient='records')


        def get_gpt_output(gpt_response):
            return gpt_response["choices"][0]["message"]["content"]


        output_df['gpt-answer'] = output_df[2].apply(get_gpt_output)


        def get_gpt_messages(gpt_response):
            return gpt_response["messages"][0]['content']


        output_df['gpt-message'] = output_df[1].apply(get_gpt_messages)
        output_df['example_id'] = output_df[0]
        output_df['prompt_tokens'] = output_df[2].apply(lambda x: x['usage']['prompt_tokens'])
        output_df['completion_tokens'] = output_df[2].apply(lambda x: x['usage']['completion_tokens'])
        output_df = output_df[['example_id', 'gpt-answer', 'gpt-message', 'prompt_tokens', 'completion_tokens']]
        combined_df = input_df.merge(output_df, on='example_id')
        combined_df.to_json(f'{experiment_name}_{gpt_model}_temp{temperature}.jsonl', lines=True,
                            orient='records')

    else:


        # parse command line arguments
        # or set the defaults here and forget about them
        parser = argparse.ArgumentParser()
        #parser.add_argument("--requests_filepath")
        #parser.add_argument("--save_filepath", default=None)
        parser.add_argument("--request_url", default="https://api.openai.com/v1/chat/completions")
        parser.add_argument("--api_key", default='os.getenv("OPENAI_API_KEY"))') #default=os.getenv("OPENAI_API_KEY"))
        parser.add_argument("--max_requests_per_minute", type=int, default=10000 * 0.9)
        parser.add_argument("--max_tokens_per_minute", type=int, default=300000 * 0.9)
        parser.add_argument("--token_encoding_name", default="cl100k_base")
        parser.add_argument("--max_attempts", type=int, default=30)
        parser.add_argument("--logging_level", default=logging.INFO)
        args = parser.parse_args()


        gpt_model = "gpt-4-0314"
        temperature = 1.0

        experiment_name = f'LIAR_new_decomposition'
        base_path = './LIAR-New'


        input_df = pd.read_json(f'{base_path}/LIAR_new.jsonl', lines=True)

        input_df['example_id'] = input_df.index # index usually easiest, but in theory can be anything unique, e.g. the input text itself if that won't be duplicated

        # this bit is for restarting if stopped partway through, e.g. cluster job timeout
        save_filepath = f'{base_path}/{experiment_name}_{gpt_model}_temp{temperature}_rawoutput.jsonl'
        if os.path.exists(save_filepath):
            reload_df = pd.read_json(save_filepath, lines=True, orient='records')
            input_df = input_df[~input_df.example_id.isin(reload_df[0])]
        # print('input_df after reload', input_df.shape)

        # this function to set up the prompt
        def text_to_messages(row):
            text = row['text']

            binary_score = f'Rate the truthfulness of the following statement: “{text}” \
                           Provide a number 0 or 1, where 0 represents falseand 1 represents true. Do not provide any explanations, only respond with the number.'

            # score_then_explain
            tmp_prompt = f'Rate the truthfulness of the following statement: "{text}" \
                         Provide a score from 0 to 100, where 0 represents definitively false and 100 represents definitively true. Then write a vertical bar "|", followed by explaining your reasoning for the score.'

            paper_prompt = f'Rate the truthfulness of the following statement: "{text}". \
             Your task is to provide a score from 0 to 100, where 0 represents definitively false and 100 represents definitively true, but you must not state your score until you have presented a thorough analysis. Do not begin your response with a number. First write your analysis, then write a vertical bar "|", then finally state your score.'

            uncertainty_paper_prompt = f'Rate the uncertainty of the following statement: "{text}". \
                         Your task is to provide a score from 0 to 100, where 0 represents definitively uncertainty and 100 represents definitively certain, but you must not state your score until you have presented a thorough analysis. Do not begin your response with a number. First write your analysis, then write a vertical bar "|", then finally state your score.'

            concise_prompt = f'Rate the truthfulness of the following statement on a 0-100 scale, where 0 represents definitively false and 100 represents definitively true: "{text}” First, provide a Chain of Thoughts analysis. Then, provide an overall uncertainty percent score enclosed by vertical lines “|”.  Finally, you are required to state your truthfulness score in squared brackets “[]”, even if you’re uncertain of your answer.'

            concise_alone = f'Rate the truthfulness of the following statement on a 0-100 scale, where 0 represents definitively false and 100 represents definitively true: "{text}” First, provide a Chain of Thoughts analysis. Then, state your truthfulness score in squared brackets “[]”, even if you’re uncertain of your answer.'

            categorical_prompt = f'Rate the truthfulness of the following statement: "{text}" using the following options: (A) Correct (B) Incorrect, (C) Somewhat true/Somewhat false. Do not begin your response with a number. First, write a thorough analysis, then write a vertical bar "|", then finally state your score.'
            #categorical_prompt = f'Label the truthfulness of the following statement into ‘true’, ‘somewhat true/false’ or ‘false’: "{text}” . First, provide a Chain of Thoughts analysis. Then, provide an overall uncertainty percent score enclosed by vertical lines “|”.  Finally, you are required to state your truthfulness label (‘true’, ‘somewhat true/false’ or ‘false’) in squared brackets “[]”, even if you’re uncertain of your answer.'

            pants_on_fire_prompt = f'Label the truthfulness of the following statement: {text}. First, provide a chain of thoughts analysis of your answer. Finally, label the truthfulness of the statement in squared brackets “[]” using PolitiFact’s Truth-O-Meter scale, where 0 represents ‘pants on fire’; 1 ‘false’; 2 ‘mostly false’; 3 ‘half true’; 4 ‘mostly true’ and 5 ‘true’.'

            politifact_context_prompt = f'Rate the following statement using PolitiFact’s Truth-O-Meter scale: "{text}". \
                    Recall the Truth-O-Meter scale: 5 = TRUE - The statement is accurate and there’s nothing significant missing. \
                    4 = MOSTLY TRUE - The statement is accurate but needs clarification or additional information. \
                    3 = HALF TRUE – The statement is partially accurate but leaves out important details or takes things out of context. \
                    2 = MOSTLY FALSE – The statement contains an element of truth but ignores critical facts that would give a different impression. \
                    1 = FALSE – The statement is not accurate. \
                    0 = PANTS ON FIRE – The statement is not accurate and makes a ridiculous claim. \
                    Do not begin your response with a number. First, write a Chain of Thoughts analysis, then write a vertical bar "|", then finally state your score.'

            strategy_qa = f'Answer the following question: "{text}" \
                Your task is to provide a truthfulness score from 0 to 100, where 0 represents definitely false and 100 represents definitely true, but you must not state your score until you have presented a thorough analysis. First, write your analysis. Then, state your 0-100 truthfulness score in squared brackets “[]”'

            strategy_qa_decomposition = f'Answer the following question by decomposing the question into subquestions and resolving them: “{text}“ \
	            Your task is to provide a truthfulness score from 0 to 100, where 0 represents definitely false and 100 represents definitely true, but you must not state your score until you have decomposed the question similar to the following examples: \
                    \
                    Question 1: Is Miami a city on the American West Coast? \
                    Decomposition 1: \
                    (A) What state is Miami located in? \
                    => Miami is a city in the state of Florida \
                    (B) Which states are part of the American West Coast? \
                    => California, Oregon and Washington \
                    (C) Is (A) included in (B) \
                    => No \
                    Answer 1 : Miami is not a city part of the American West Coast. [0] \
                    \
                    Question 2: Was Bruce Lee absent from the 1964 University of Washington graduation ceremony? \
                    Decomposition 2: \
                    (A) When did the University of Washington graduation ceremony for the class of 1964 take place? \
                    => Summer 1964 \
                    (B) What college did Bruce Lee attend? \
                    => University of Washington \
                    (C) Did Bruce Lee drop out of the University of Washington? \
                    => Yes, he drop out in Spring 1964 \
                    Did (A) occur after (C)? \
                    => Yes. \
                        Answer 2 : Bruce Lee did not complete his studies at University of Washington, so he did not attend the graduation ceremony for the 1964 class. [100] \
                        \
                First, write your decomposition analysis. Then, state your 0-100 truthfulness score in squared brackets “[]”.'
        
            LIAR_decomposition = f'Rate the truthfulness of  the following statement by decomposing the statement into subquestions and resolving them: “{text}“ \
	            Your task is to provide a truthfulness score from 0 to 100, where 0 represents definitely false and 100 represents definitely true, but you must not state your score until you have decomposed the question similar to the following examples: \
                    \
                    Statement 1: Miami a city on the American West Coast. \
                    Decomposition 1: \
                    (A) What state is Miami located in? \
                    => Miami is a city in the state of Florida \
                    (B) Which states are part of the American West Coast? \
                    => California, Oregon and Washington \
                    (C) Is (A) included in (B) \
                    => No \
                    Answer 1 : Miami is not a city part of the American West Coast. [0] \
                    \
                    Statement 2: Bruce Lee absent from the 1964 University of Washington graduation ceremony. \
                    Decomposition 2: \
                    (A) When did the University of Washington graduation ceremony for the class of 1964 take place? \
                    => Summer 1964 \
                    (B) What college did Bruce Lee attend? \
                    => University of Washington \
                    (C) Did Bruce Lee drop out of the University of Washington? \
                    => Yes, he drop out in Spring 1964 \
                    Did (A) occur after (C)? \
                    => Yes. \
                        Answer 2 : Bruce Lee did not complete his studies at University of Washington, so he did not attend the graduation ceremony for the 1964 class. [100] \
                        \
                First, write your decomposition analysis. Then, state your 0-100 truthfulness score in squared brackets “[]”.'

            LIAR_new_decomp = f'Rate the truthfulness of the following statement by decomposing the statement into sub questions and resolving them: “{text}“ \
                Your task is to provide a truthfulness score from 0 to 100 enclosed by square brackets “[]”, where 0 represents definitely false and 100 represents definitely true, for each subquestion-resolution pair. Finally, you must state your final analysis and truthfulness score enclosed by vertical lines “| |”. Please follow the same format as the following examples: \
                    \
                    Example #1: Miami is a city on the American West Coast. \
                    Decompositions: \
                    (A) What state is Miami located in? \
                        => Miami is a city in the state of Florida [100] \
                    (B) Which states are part of the American West Coast? \
                        => California, Oregon and Washington [100] \
                    Final answer : Miami, located in the state of Florida, is not a city part of the American West Coast. |0| \
                    \
                    Example #2: A silicon shortage would be bad for Intel\'s sales. \
                    Decompositions: \
                    (A) What kind of products does Intel make? \
                        => Intel is a multinational company that produces semiconductor chips. [100] \
                    (B) Is silicon a crucial component for Intel\'s products? \
                        => Yes, silicon is a fundamental material for semiconductor chips, and Intel heavily relies on it. [100] \
                    (C) Would a reduction in chip production negatively affect Intel\'s sales? \
                        => Yes, as chip shortages could lead to a decrease in product availability and potentially impact sales. [70] \
                    Final answer: Based on this analysis, it can be concluded that a silicon shortage would have a negative impact on Intel\'s sales due to its dependence on silicon for chip production. |90|'
            

            return [{"role": "user", "content": LIAR_new_decomp}]

        input_df['messages'] = input_df.apply(lambda row: text_to_messages(row), axis=1)
        input_df['temperature'] = temperature
        input_df['model'] = gpt_model
        input_df['max_tokens'] = 1000 # maximum tokens in the response

        requests_filepath = f'{base_path}/{experiment_name}_{gpt_model}_temp{temperature}_preprocessedforparallel.jsonl'
        input_df[['example_id','messages', 'temperature','model', 'max_tokens']].to_json(requests_filepath, lines=True, orient='records') # only fields gpt is expecting + example_id
        input_df.to_json(requests_filepath.split('.jsonl')[0] + 'full.jsonl', lines=True, orient='records') # not really needed, just record of the full input df

        # run script
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=requests_filepath,
                save_filepath=save_filepath,
                request_url=args.request_url,
                api_key=args.api_key,
                max_requests_per_minute=float(args.max_requests_per_minute),
                max_tokens_per_minute=float(args.max_tokens_per_minute),
                token_encoding_name=args.token_encoding_name,
                max_attempts=int(args.max_attempts),
                logging_level=int(args.logging_level),
            )
        )

        # match things back together with example_id, and also get rid of unneeded metadata in the gpt response
        output_df = pd.read_json(save_filepath, lines=True, orient='records')
        def get_gpt_output(gpt_response):
            return gpt_response["choices"][0]["message"]["content"]
        output_df['gpt-answer'] = output_df[2].apply(get_gpt_output)
        def get_gpt_messages(gpt_response):
            return gpt_response["messages"][0]['content']
        output_df['gpt-message'] = output_df[1].apply(get_gpt_messages)
        output_df['example_id'] = output_df[0]
        output_df['prompt_tokens'] = output_df[2].apply(lambda x: x['usage']['prompt_tokens'])
        output_df['completion_tokens'] = output_df[2].apply(lambda x: x['usage']['completion_tokens'])
        output_df = output_df[['example_id','gpt-answer', 'gpt-message', 'prompt_tokens', 'completion_tokens']]
        combined_df = input_df.merge(output_df, on='example_id')
        combined_df.to_json(f'{base_path}/{experiment_name}_{gpt_model}_temp{temperature}.jsonl', lines=True, orient='records')

    #import subprocess

    #result = subprocess.run(['python', 'evaluate.py', '--experiment_name', experiment_name], capture_output=True, text=True)

    #print(result.stdout)




