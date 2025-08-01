from transformers import AutoTokenizer, AutoModel
import torch
import json
import os
from typing import Dict, Any, Optional, Tuple


class LLMResponseEncoder:
    def __init__(self, model_name: str = "./pretrained_models/sup-simcse-roberta-large"):
        """
        Initialize encoder

        Args:
            model_name: pretrained model name or local path
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Load tokenizer and model
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode_texts(self, texts: list, batch_size: int = 64) -> torch.Tensor:
        """
        Encode a list of texts

        Args:
            texts: list of texts
            batch_size: batch size

        Returns:
            Tensor of embeddings
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            print(f"Processing batch: {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

            all_embeddings.append(embeddings.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings

    def load_and_encode_profiles(self, dataset_name: str, batch_size: int = 64) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Load and encode user and item profiles

        Args:
            dataset_name: name of the dataset
            batch_size: batch size for encoding

        Returns:
            (user embeddings tensor, item embeddings tensor)
        """
        base_input_dir = f"./batch_output/{dataset_name}"

        user_embeddings = None
        item_embeddings = None

        user_file = os.path.join(base_input_dir, "llm_comprehension_user1.jsonl")
        if os.path.exists(user_file):
            print(f"Processing user profiles: {user_file}")
            user_texts, user_ids = self._load_profiles_from_jsonl(user_file)
            if user_texts:
                user_embeddings = self.encode_texts(user_texts, batch_size)
                print(f"User embeddings shape: {user_embeddings.shape}")
            else:
                print("No successful user profiles found")
        else:
            print(f"User file not found: {user_file}")

        item_file = os.path.join(base_input_dir, "llm_comprehension_item1.jsonl")
        if os.path.exists(item_file):
            print(f"Processing item profiles: {item_file}")
            item_texts, item_ids = self._load_profiles_from_jsonl(item_file)
            if item_texts:
                item_embeddings = self.encode_texts(item_texts, batch_size)
                print(f"Item embeddings shape: {item_embeddings.shape}")
            else:
                print("No successful item profiles found")
        else:
            print(f"Item file not found: {item_file}")

        return user_embeddings, item_embeddings

    def _load_profiles_from_jsonl(self, file_path: str) -> Tuple[list, list]:
        """
        Load profiles from a JSONL file

        Args:
            file_path: path to JSONL file

        Returns:
            (list of texts, list of IDs)
        """
        texts = []
        ids = []
        total_lines = 0
        success_count = 0
        failed_count = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                try:
                    data = json.loads(line.strip())
                    if data.get('success', False) and data.get('profile'):
                        texts.append(data['profile'])
                        ids.append(data['center_id'])
                        success_count += 1
                    else:
                        failed_count += 1
                        print(f"Skipping line {line_num}: success={data.get('success', False)}, has_profile={bool(data.get('profile'))}, center_id={data.get('center_id', None)}")
                except json.JSONDecodeError as e:
                    failed_count += 1
                    print(f"JSON parsing error on line {line_num}: {e}")

        print(f"Finished processing file {file_path}:")
        print(f"  Total lines: {total_lines}")
        print(f"  Loaded successfully: {success_count}")
        print(f"  Skipped/failed: {failed_count}")

        return texts, ids

    def save_embeddings(self, user_embeddings: Optional[torch.Tensor], item_embeddings: Optional[torch.Tensor], dataset_name: str) -> None:
        """
        Save embeddings to files

        Args:
            user_embeddings: tensor of user embeddings
            item_embeddings: tensor of item embeddings
            dataset_name: name of the dataset
        """
        output_dir = f"./data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        if user_embeddings is not None:
            user_embedding_file = os.path.join(output_dir, "user_embeddings.pt")
            torch.save(user_embeddings, user_embedding_file)
            print(f"User embeddings saved to: {user_embedding_file}")

        if item_embeddings is not None:
            item_embedding_file = os.path.join(output_dir, "item_embeddings.pt")
            torch.save(item_embeddings, item_embedding_file)
            print(f"Item embeddings saved to: {item_embedding_file}")

    def load_embeddings(self, dataset_name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Load saved embeddings from files

        Args:
            dataset_name: name of the dataset

        Returns:
            (user embeddings tensor, item embeddings tensor)
        """
        base_dir = f"./data/{dataset_name}"
        user_embeddings = None
        item_embeddings = None

        user_file = os.path.join(base_dir, "user_embeddings.pt")
        if os.path.exists(user_file):
            user_embeddings = torch.load(user_file)
            print(f"Loaded user embeddings: {user_embeddings.shape}")

        item_file = os.path.join(base_dir, "item_embeddings.pt")
        if os.path.exists(item_file):
            item_embeddings = torch.load(item_file)
            print(f"Loaded item embeddings: {item_embeddings.shape}")

        return user_embeddings, item_embeddings


def main():
    """Main function example"""
    # Initialize encoder
    encoder = LLMResponseEncoder()

    # Dataset name
    dataset_name = "dbbook2014"  # Replace with your dataset name

    # Load and encode profiles
    print(f"Starting processing dataset: {dataset_name}")
    user_embeddings, item_embeddings = encoder.load_and_encode_profiles(dataset_name, batch_size=32)

    # Save embeddings
    if user_embeddings is not None or item_embeddings is not None:
        encoder.save_embeddings(user_embeddings, item_embeddings, dataset_name)
        print("Encoding completed!")

        # Verify loading
        print("\nVerifying loading...")
        loaded_user_emb, loaded_item_emb = encoder.load_embeddings(dataset_name)

        if loaded_user_emb is not None:
            print(f"User embeddings shape: {loaded_user_emb.shape}")

        if loaded_item_emb is not None:
            print(f"Item embeddings shape: {loaded_item_emb.shape}")
    else:
        print("No data found to encode")


if __name__ == "__main__":
    main()
