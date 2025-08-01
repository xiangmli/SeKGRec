import json
import os
import time
from typing import Dict, Any, Optional
from pathlib import Path


import os
import json
import time
from typing import Optional

class LLMProfileGenerator:
    def __init__(self, api_key: str, base_url: str = ""):
        """
        Initialize the LLM profile generator

        Args:
            api_key: API key
            base_url: Base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url

    def call_llm(self, messages: list, max_tokens: int = 500, temperature: float = 0.3):
        # Replace with your own LLM call implementation
        return 0

    def process_jsonl_file(self, input_file: str, output_file: str, delay: float = 1.0,
                           save_interval: int = 10, target_ids: Optional[set] = None) -> None:
        """
        Process a JSONL file and generate LLM responses for each entry

        Args:
            input_file: Path to the input file
            output_file: Path to the output file
            delay: Delay between requests (seconds)
            save_interval: Number of records to process before saving
            target_ids: Set of IDs to process; if None, all entries are processed
        """
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        processed_count = 0
        success_count = 0
        skipped_count = 0
        batch_results = []

        # Open output file in append mode if it already exists
        file_mode = 'a' if os.path.exists(output_file) else 'w'

        if target_ids is not None:
            print(f"Processing {len(target_ids)} target IDs: {sorted(list(target_ids))}")

        with open(input_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                try:
                    data = json.loads(line.strip())
                    center_id = data.get('center_id')
                    center_name = data.get('center_name')
                    node_type = data.get('node_type')
                    messages = data.get('messages', [])

                    if target_ids is not None and center_id not in target_ids:
                        skipped_count += 1
                        if skipped_count % 100 == 0:
                            print(f"Skipped {skipped_count} records not in target list")
                        continue

                    print(f"Processing record {line_num}: {center_name} ({node_type}), ID: {center_id}")

                    llm_response = self.call_llm(messages)

                    result = {
                        'center_id': center_id,
                        'center_name': center_name,
                        'node_type': node_type,
                        'profile': llm_response,
                        'success': llm_response is not None
                    }

                    batch_results.append(result)

                    if llm_response is not None:
                        success_count += 1
                        print("✓ Profile generated successfully")
                    else:
                        print("✗ Generation failed")

                    processed_count += 1

                    if len(batch_results) >= save_interval:
                        self._save_batch_results(output_file, batch_results, file_mode)
                        batch_results = []
                        file_mode = 'a'
                        print(f"Saved {processed_count} records")

                    if delay > 0:
                        time.sleep(delay)

                except json.JSONDecodeError as e:
                    print(f"JSON parsing error on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue

            if batch_results:
                self._save_batch_results(output_file, batch_results, file_mode)
                print(f"Saved remaining {len(batch_results)} records")

        print("\nProcessing completed:")
        print(f"  Total records processed: {processed_count}")
        print(f"  Successful generations: {success_count}")
        if target_ids is not None:
            print(f"  Skipped records: {skipped_count}")

    def _save_batch_results(self, output_file: str, batch_results: list, file_mode: str) -> None:
        """
        Save a batch of results to file

        Args:
            output_file: Path to the output file
            batch_results: List of batch result dictionaries
            file_mode: File open mode ('w' or 'a')
        """
        with open(output_file, file_mode, encoding='utf-8') as outfile:
            for result in batch_results:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            outfile.flush()


    def process_dataset(self, dataset_name: str, delay: float = 1.0, save_interval: int = 10,
                       target_user_ids: Optional[list] = None, target_item_ids: Optional[list] = None) -> None:
        """
        Process the user and item subgraphs for the entire dataset

        Args:
            dataset_name: Dataset name
            delay: Request delay time (seconds)
            save_interval: Number of records processed before saving
            target_user_ids: List of user IDs to process, if None all users are processed
            target_item_ids: List of item IDs to process, if None all items are processed
        """
        base_input_dir = f"./batch_input/{dataset_name}"
        base_output_dir = f"./batch_output/{dataset_name}"

        target_user_set = set(map(str, target_user_ids)) if target_user_ids else None
        target_item_set = set(map(str, target_item_ids)) if target_item_ids else None

        user_input_file = os.path.join(base_input_dir, "user_subgraph_llm_input.jsonl")
        user_output_file = os.path.join(base_output_dir, "llm_comprehension_user.jsonl")

        if os.path.exists(user_input_file):
            if target_user_set:
                self.process_jsonl_file(user_input_file, user_output_file, delay, save_interval, target_user_set)
        else:
            print(f"User subgraph file does not exist: {user_input_file}")

        item_input_file = os.path.join(base_input_dir, "item_subgraph_llm_input.jsonl")
        item_output_file = os.path.join(base_output_dir, "llm_comprehension_item.jsonl")

        if os.path.exists(item_input_file):
            if target_item_set:
                self.process_jsonl_file(item_input_file, item_output_file, delay, save_interval, target_item_set)
        else:
            print(f"Item subgraph file does not exist: {item_input_file}")

    def process_specific_users(self, dataset_name: str, user_ids: list, delay: float = 1.0,
                              save_interval: int = 10) -> None:
        """
        Convenient method to process specific user IDs

        Args:
            dataset_name: Dataset name
            user_ids: List of user IDs
            delay: Request delay time (seconds)
            save_interval: Number of records processed before saving
        """
        self.process_dataset(dataset_name, delay, save_interval, target_user_ids=user_ids)

    def process_specific_items(self, dataset_name: str, item_ids: list, delay: float = 1.0,
                              save_interval: int = 10) -> None:
        """
        Convenient method to process specific item IDs

        Args:
            dataset_name: Dataset name
            item_ids: List of item IDs
            delay: Request delay time (seconds)
            save_interval: Number of records processed before saving
        """
        self.process_dataset(dataset_name, delay, save_interval, target_item_ids=item_ids)

    def load_profiles(self, dataset_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Load the generated configuration files

        Args:
            dataset_name: Dataset name

        Returns:
            A dictionary containing user and item profiles
        """
        profiles = {
            'users': {},
            'items': {}
        }

        base_output_dir = f"./V0.1/batch_output/{dataset_name}"

        user_file = os.path.join(base_output_dir, "llm_comprehension_user.jsonl")
        if os.path.exists(user_file):
            with open(user_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data['success']:
                        profiles['users'][data['center_id']] = {
                            'name': data['center_name'],
                            'profile': data['profile']
                        }

        item_file = os.path.join(base_output_dir, "llm_comprehension_item.jsonl")
        if os.path.exists(item_file):
            with open(item_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data['success']:
                        profiles['items'][data['center_id']] = {
                            'name': data['center_name'],
                            'profile': data['profile']
                        }

        return profiles



def main():
    API_KEY = ""

    generator = LLMProfileGenerator(API_KEY)

    dataset_name = "dbbook2014"

    generator.process_dataset(dataset_name, delay=1.0, save_interval=10)


if __name__ == "__main__":
    main()




