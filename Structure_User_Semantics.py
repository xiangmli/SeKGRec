import json
import jsonlines
import os
import openai
import json
from openai import OpenAI
import httpx
from tqdm import tqdm
import time
import pandas as pd


def parse_merge_output(merge_output_content):
    try:
        content = merge_output_content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()

        merge_data = json.loads(content)
        return merge_data
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        show_json_error_context(content, e)
        return None


def show_json_error_context(content, error, context_lines=3, context_chars=50):
    try:
        error_pos = error.pos
        error_line = error.lineno
        error_col = error.colno

        print(f"\n❌ JSON parsing error details:")
        print(f"Error: {error.msg}")
        print(f"Location: Line {error_line}, Column {error_col}, Character {error_pos}")

        lines = content.split('\n')

        print(f"\n📍 Error context:")

        start_line = max(0, error_line - context_lines - 1)
        end_line = min(len(lines), error_line + context_lines)

        for i in range(start_line, end_line):
            line_num = i + 1
            line_content = lines[i]

            if line_num == error_line:
                print(f">>> {line_num:3d}: {line_content}")
                pointer = " " * (8 + error_col - 1) + "^"
                print(f"    {pointer} Here is the issue!")
            else:
                print(f"    {line_num:3d}: {line_content}")

        if len(lines) == 1 or error_pos < len(content):
            print(f"\n🔍 Character-level context:")
            start_char = max(0, error_pos - context_chars)
            end_char = min(len(content), error_pos + context_chars)

            context_str = content[start_char:end_char]
            error_char_in_context = error_pos - start_char

            print(f"Content snippet: '{context_str}'")
            print(f"Error position:  {' ' * error_char_in_context}^")

            if error_pos < len(content):
                error_char = content[error_pos]
                print(f"Error character: '{error_char}' (ASCII: {ord(error_char)})")

    except Exception as e:
        print(f"Error occurred while displaying error context: {e}")
        if hasattr(error, 'pos') and error.pos < len(content):
            start = max(0, error.pos - 100)
            end = min(len(content), error.pos + 100)
            print(f"\nNearby content:")
            print(f"'{content[start:end]}'")

def build_merged_intent_triples(intent_triples, output_dict, merge_output_path):
    cluster_merge_mapping = {}

    with open(merge_output_path, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            cluster_id = line['custom_id']
            raw_content = line['response']['body']['choices'][0]['message']['content']
            merge_data = parse_merge_output(raw_content)
            if merge_data:
                cluster_merge_mapping[cluster_id] = merge_data

    original_to_merged = {}

    for cluster_id, merge_data in cluster_merge_mapping.items():
        if 'merged_intents' in merge_data:
            for merged_group in merge_data['merged_intents']:
                representative_intent = merged_group['representative_intent']
                merged_from = merged_group['merged_from']

                for original_intent in merged_from:
                    original_to_merged[original_intent] = representative_intent

        if 'unchanged_intents' in merge_data:
            for unchanged_intent in merge_data['unchanged_intents']:
                original_to_merged[unchanged_intent] = unchanged_intent

    merged_intent_triples = []

    for user_id, original_intent in intent_triples:
        merged_intent = original_to_merged.get(original_intent, original_intent)
        merged_intent_triples.append([user_id, merged_intent])

    return merged_intent_triples

def load_saved_data(dataset, llm):
    user_intent_file = f'User_Intent/{dataset}_user_intent_{llm}.json'
    with open(user_intent_file, 'r', encoding='utf-8') as f:
        intent_triples = json.load(f)

    intent_cluster_file = f'User_Intent/{dataset}_intent_cluster_{llm}.json'
    with open(intent_cluster_file, 'r', encoding='utf-8') as f:
        output_dict = json.load(f)

    return intent_triples, output_dict


def get_system_prompt_second_merge():
    return """You are an expert in intent analysis and text processing. Your task is to perform a CONSERVATIVE second-pass merge of intents that are highly similar.

Instructions:
1. Only merge intents that are virtually identical in meaning and purpose.
2. Be VERY selective - only merge when intents are clearly duplicates or near-duplicates.
3. Different wording of the same core intent can be merged.
4. Keep intents separate if there's any meaningful difference in scope or context.
5. When in doubt, DO NOT merge - err on the side of keeping intents separate.
6. Output must be in strict JSON format.

Output format:
{
    "merged_intents": [
        {
            "representative_intent": "chosen representative intent",
            "merged_from": ["intent1", "intent2", "intent3"]
        }
    ],
    "unchanged_intents": ["unique intent1", "unique intent2"]
}
"""


def get_user_prompt_second_merge(all_merged_intents):
    intents_str = '\n'.join([f"- {intent}" for intent in all_merged_intents])

    prompt = f"""Please perform a conservative second-pass analysis of the following merged intents. Only merge intents that are virtually identical in meaning:

{intents_str}

IMPORTANT: Be very conservative. Only merge intents that are essentially the same thing expressed differently. If there's any meaningful difference, keep them separate."""

    return prompt


def extract_unique_merged_intents(merged_intent_triples):
    unique_intents = set()
    for user_id, merged_intent in merged_intent_triples:
        unique_intents.add(merged_intent)

    return sorted(list(unique_intents))

def prepare_second_merge_input(merged_intent_triples, dataset, llm):
    all_merged_intents = extract_unique_merged_intents(merged_intent_triples)

    message = [
        {"role": "system", "content": get_system_prompt_second_merge()},
        {"role": "user", "content": get_user_prompt_second_merge(all_merged_intents)}
    ]

    row = {
        "custom_id": "second_merge_all_intents",
        "method": "POST",
        "url": "",
        "body": {
            "model": llm,
            "messages": message,
            "max_tokens": 10000,
            "temperature": 0.05
        }
    }

    os.makedirs('batch_input', exist_ok=True)

    output_file = f'batch_input/{dataset}_second_intent_merge_{llm}_input.jsonl'
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write(row)

    return output_file

client = OpenAI()


def test_api_connection():
    try:
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            max_tokens=10
        )
        return True
    except Exception as e:
        return False

def call_llm_single(message, model="gpt-3.5-turbo", max_tokens=1000):
    # replace with your own call llm method
    return 0

def process_second_merge_batch(dataset, llm):

    input_file = f'batch_input/{dataset}_second_intent_merge_{llm}_input.jsonl'
    output_file = f'batch_output/{dataset}_second_intent_merge_{llm}_output.jsonl'

    if not test_api_connection():
        return False

    output_content = []
    success_count = 0
    error_count = 0

    if not os.path.exists(input_file):
        return False

    try:
        with jsonlines.open(input_file, 'r') as reader:
            total_requests = sum(1 for _ in reader)
    except Exception as e:
        return False

    os.makedirs('batch_output', exist_ok=True)

    with jsonlines.open(input_file, 'r') as reader:
        for i, request in enumerate(tqdm(reader, total=total_requests, desc='Processing Second Merge Requests')):
            custom_id = request.get('custom_id', f'unknown_{i}')

            try:
                messages = request['body']['messages']
                model = request['body']['model']
                max_tokens = request['body']['max_tokens']

                if model == "gpt-3.5-turbo-0125":
                    model = "gpt-3.5-turbo"

                response_content = call_llm_single(messages, model, max_tokens)

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

                    try:
                        import json
                        parsed_response = json.loads(response_content)
                        if 'merged_intents' in parsed_response or 'unchanged_intents' in parsed_response:
                            pass
                        else:
                            pass
                    except json.JSONDecodeError:
                        pass

                else:
                    error_count += 1

                    output_row = {
                        "id": f"batch_req_{custom_id}",
                        "custom_id": custom_id,
                        "response": None,
                        "error": {
                            "code": "api_error",
                            "message": "Failed to get response from LLM for second merge"
                        }
                    }

                output_content.append(output_row)

                temp_output = f"{output_file}.temp"
                with jsonlines.open(temp_output, 'w') as writer:
                    for row in output_content:
                        writer.write(row)

                time.sleep(2)

            except KeyError as e:
                error_count += 1
            except Exception as e:
                error_count += 1

    try:
        with jsonlines.open(output_file, 'w') as writer:
            for row in output_content:
                writer.write(row)

        temp_file = f"{output_file}.temp"
        if os.path.exists(temp_file):
            os.remove(temp_file)

        return success_count > 0

    except Exception as e:
        return False


def build_final_merged_intent_triples(merged_intent_triples, second_merge_output_path):

    first_to_final_mapping = {}

    with open(second_merge_output_path, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            raw_content = line['response']['body']['choices'][0]['message']['content']
            merge_data = parse_merge_output(raw_content)

            if merge_data:
                if 'merged_intents' in merge_data:
                    for merged_group in merge_data['merged_intents']:
                        representative_intent = merged_group['representative_intent']
                        merged_from = merged_group['merged_from']

                        for first_merged_intent in merged_from:
                            first_to_final_mapping[first_merged_intent] = representative_intent

                if 'unchanged_intents' in merge_data:
                    for unchanged_intent in merge_data['unchanged_intents']:
                        first_to_final_mapping[unchanged_intent] = unchanged_intent

    user_intent_seen = set()
    final_intent_triples = []

    for user_id, first_merged_intent in merged_intent_triples:
        final_merged_intent = first_to_final_mapping.get(first_merged_intent, first_merged_intent)

        user_intent_pair = (user_id, final_merged_intent)
        if user_intent_pair not in user_intent_seen:
            user_intent_seen.add(user_intent_pair)
            final_intent_triples.append([user_id, final_merged_intent])

    return final_intent_triples

def get_entity_max():
    data_path = f'./data/{dataset}'
    if dataset in ['dbbook2014', 'ml1m', 'test_dataset']:
        item_list = pd.read_csv(data_path + '/item_list.txt', sep=' ')
        entity_list = pd.read_csv(data_path + '/entity_list.txt', sep=' ')
    else:
        item_list = list()
        entity_list = list()
        lines = open(data_path + '/item_list.txt', 'r').readlines()
        for l in lines[1:]:
            if l == "\n":
                continue
            tmps = l.replace("\n", "").strip()
            elems = tmps.split(' ')
            org_id = elems[0]
            remap_id = elems[1]
            if len(elems[2:]) == 0:
                continue
            title = ' '.join(elems[2:])
            item_list.append([org_id, remap_id, title])
        item_list = pd.DataFrame(item_list, columns=['org_id', 'remap_id', 'entity_name'])
        lines = open(data_path + '/entity_list.txt', 'r').readlines()
        for l in lines[1:]:
            if l == "\n":
                continue
            tmps = l.replace("\n", "").strip()
            elems = tmps.split()
            org_id = elems[0]
            remap_id = elems[1]
            entity_list.append([org_id, remap_id])
        entity_list = pd.DataFrame(entity_list, columns=['org_id', 'remap_id'])

    entity_max = entity_list['remap_id'].max() + 1
    return entity_max

def create_intent_mapping_and_save(final_intent_triples, entity_max, output_dir,
                                   filter_sparse_dense=False, min_users=2, max_user_ratio=0.2):

    df = pd.DataFrame(final_intent_triples, columns=['user_id', 'intent_text'])

    if filter_sparse_dense:
        intent_user_count = df.groupby('intent_text')['user_id'].nunique().reset_index()
        intent_user_count.columns = ['intent_text', 'user_count']
        intent_user_count = intent_user_count.sort_values('user_count', ascending=False)

        total_users = df['user_id'].nunique()
        max_users = int(total_users * max_user_ratio)

        sparse_intents = intent_user_count[intent_user_count['user_count'] < min_users]['intent_text'].tolist()
        dense_intents = intent_user_count[intent_user_count['user_count'] > max_users]['intent_text'].tolist()

        del_intents = sparse_intents + dense_intents

        df_filtered = df[~df['intent_text'].isin(del_intents)]

        final_intent_triples = df_filtered.values.tolist()

    unique_intents = set()
    for user_id, intent_text in final_intent_triples:
        unique_intents.add(intent_text)

    intent_to_id = {}
    id_to_intent = {}
    current_id = entity_max

    for intent_text in sorted(unique_intents):
        intent_to_id[intent_text] = current_id
        id_to_intent[current_id] = intent_text
        current_id += 1

    user_intent_id_pairs = []
    for user_id, intent_text in final_intent_triples:
        intent_id = intent_to_id[intent_text]
        user_intent_id_pairs.append([user_id, intent_id])

    user_intent_file = os.path.join(output_dir, 'user_intent.txt')
    with open(user_intent_file, 'w', encoding='utf-8') as f:
        for user_id, intent_id in user_intent_id_pairs:
            f.write(f"{user_id}\t{intent_id}\n")

    intent_mapping_file = os.path.join(output_dir, 'intent_list.txt')
    with open(intent_mapping_file, 'w', encoding='utf-8') as f:
        for intent_id in sorted(id_to_intent.keys()):
            intent_text = id_to_intent[intent_id]
            f.write(f"{intent_id}\t{intent_text}\n")

    return intent_to_id, id_to_intent


if __name__ == "__main__":
    dataset = 'dbbook2014'
    llm = 'gpt-3.5-turbo-0125'
    merge_output_path = f'batch_output/{dataset}_first_intent_merge_output.jsonl'

    intent_triples, output_dict = load_saved_data(dataset, llm)
    merged_intent_triples = build_merged_intent_triples(
        intent_triples,
        output_dict,
        merge_output_path
    )

    entity_max = get_entity_max()

    use_filter = False

    if use_filter:
        intent_to_id, id_to_intent = create_intent_mapping_and_save(
            merged_intent_triples,
            entity_max,
            output_dir=f"data\\{dataset}",
            filter_sparse_dense=True,
            min_users=2,
            max_user_ratio=0.2
        )
    else:
        intent_to_id, id_to_intent = create_intent_mapping_and_save(
            merged_intent_triples,
            entity_max,
            output_dir=f"data\\{dataset}",
            filter_sparse_dense=False
        )
