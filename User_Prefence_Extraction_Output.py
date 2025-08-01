import openai
import json
from openai import OpenAI
import jsonlines
from tqdm import tqdm
import time
import os
import httpx


def get_prompt_generate(history, field):
    f = field.strip('s')
    prompt = '[' + ', '.join(history) + ']'
    return prompt

def get_system_generate(history, field):
    f = field.strip('s')
    return f"You will be provided with a list of {field} an anonymous user has liked, and your task is to infer the user's interests based on the list and your extensive knowledge. List no more than five the top interests of this anonymous user. No further explanation is needed. Please use a comma to split the interests."

def call_llm_single(client, message, model="gpt-3.5-turbo", max_tokens=1000):
    return None

def test_api_connection(client):
    try:
        print("Testing API connection...")
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            max_tokens=10
        )
        print("API connection successful!")
        print(f"Test response: {test_response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"API connection test failed: {e}")
        return False

def process_batch_requests(client, input_file, output_file):
    if not test_api_connection(client):
        print("API connection issue")
        return

    output_content = []
    success_count = 0
    error_count = 0

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    try:
        with jsonlines.open(input_file, 'r') as reader:
            total_requests = sum(1 for _ in reader)
        print(f"Found {total_requests} requests")
    except Exception as e:
        print(f"Failed to read input file: {e}")
        return

    with jsonlines.open(input_file, 'r') as reader:
        for i, request in enumerate(tqdm(reader, total=total_requests, desc='Processing LLM requests')):
            custom_id = request.get('custom_id', f'unknown_{i}')

            try:
                messages = request['body']['messages']
                model = request['body']['model']
                max_tokens = request['body']['max_tokens']

                if model == "gpt-3.5-turbo-0125":
                    model = "gpt-3.5-turbo"

                print(f"\nProcessing request {custom_id} ({i + 1}/{total_requests})")

                response_content = call_llm_single(client, messages, model, max_tokens)

                if response_content:
                    success_count += 1
                    output_row = {
                        "id": f"batch_req_{custom_id}",
                        "custom_id": custom_id,
                        "response": {
                            "status_code": 200,
                            "request_id": f"req_{custom_id}",
                            "body": {
                                "id": f"chatcmpl-{custom_id}",
                                "object": "chat.completion",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": response_content
                                        },
                                        "finish_reason": "stop"
                                    }
                                ]
                            }
                        },
                        "error": None
                    }
                else:
                    error_count += 1
                    output_row = {
                        "id": f"batch_req_{custom_id}",
                        "custom_id": custom_id,
                        "response": None,
                        "error": {
                            "code": "api_error",
                            "message": "Failed to get response from LLM"
                        }
                    }

                output_content.append(output_row)

                if (i + 1) % 10 == 0:
                    temp_output = f"{output_file}.temp"
                    with jsonlines.open(temp_output, 'w') as writer:
                        for row in output_content:
                            writer.write(row)
                    print(f"Temporary save to {temp_output}")

                time.sleep(1)

            except KeyError as e:
                print(f"Missing field in request: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing request {custom_id}: {e}")
                error_count += 1

    try:
        with jsonlines.open(output_file, 'w') as writer:
            for row in output_content:
                writer.write(row)

        print(f"\nBatch processing complete!")
        print(f"Success: {success_count}, Failures: {error_count}")
        print(f"Output saved to: {output_file}")

        temp_file = f"{output_file}.temp"
        if os.path.exists(temp_file):
            os.remove(temp_file)

    except Exception as e:
        print(f"Failed to save output file: {e}")

def main():
    llm = "gpt-3.5-turbo-0125"
    version = 'v1'
    max_his_num = 30
    dataset = 'dbbook2014'

    client = OpenAI()

    input_file = f'batch_input/{dataset}_user_max_his{max_his_num}_{llm}_input.jsonl'
    output_file = f'batch_output/{dataset}_max{max_his_num}_output.jsonl'

    os.makedirs('batch_output', exist_ok=True)

    print(f"Starting batch processing from {input_file}")
    process_batch_requests(client, input_file, output_file)

if __name__ == "__main__":
    main()
