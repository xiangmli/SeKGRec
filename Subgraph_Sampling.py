import numpy as np
import pandas as pd
import torch
import time
import json
from tqdm import tqdm
from torch_geometric.data import Data
import argparse


from utils import MyLoader, init_seed, generate_kg_batch
from evaluate import eval_model
from prettytable import PrettyTable
from torch_geometric.utils import coalesce
from collections import defaultdict
from typing import List, Tuple, Dict, Any
import openai
from openai import OpenAI
import httpx
import os
import requests
def parse_args():
    parser = argparse.ArgumentParser(description="Run CIKGRec.")
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed.')
    parser.add_argument('--dataset', nargs='?', default='dbbook2014',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    args = parser.parse_args()
    return args


def build_graph(loader, inverse=True):
    if inverse:
        all_heads = []
        all_tails = []
        all_relations = []

        all_heads.extend(loader.kg['head'].to_list())
        all_tails.extend(loader.kg['tail'].to_list())
        all_relations.extend(loader.kg['relation'].to_list())

        all_heads.extend(loader.kg['tail'].to_list())
        all_tails.extend(loader.kg['head'].to_list())
        all_relations.extend(loader.kg['relation'].to_list())
    else:
        all_heads = loader.kg['head'].to_list()
        all_tails = loader.kg['tail'].to_list()
        all_relations = loader.kg['relation'].to_list()

    edge_index = [all_heads, all_tails]
    edge_index = torch.LongTensor(edge_index)

    edge_attr = torch.LongTensor(all_relations)

    edge_index, edge_attr = coalesce(edge_index, edge_attr)

    graph = Data(edge_index=edge_index.contiguous(), edge_attr=edge_attr.contiguous())

    return graph


class LLMGraphSampler:
    def __init__(self, loader, model="gpt-3.5-turbo", max_hops=3):
        self.loader = loader
        self.model = model
        self.max_hops = max_hops

        self.build_node_mappings()

    def process_name(self, name):
        if name.startswith('http://dbpedia.org/resource/'):
            name = name.replace('http://dbpedia.org/resource/', '')
            name = name.replace('_', ' ')

        return name

    def load_item_name_mapping(self, dataset_path):
        item_name_mapping = {}
        item_list_file = f'{dataset_path}/item_list.txt'

        try:
            with open(item_list_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(' ')
                    if len(parts) >= 3:
                        try:
                            original_id = int(parts[1])
                            item_name = parts[2]

                            item_name = self.process_name(item_name)

                            if item_name:
                                item_name_mapping[original_id] = item_name

                        except ValueError:
                            continue

        except FileNotFoundError:
            pass
        except Exception as e:
            pass

        return item_name_mapping

    def load_entity_name_mapping(self, dataset_path):
        entity_name_mapping = {}
        entity_list_file = f'{dataset_path}/entity_list.txt'

        try:
            with open(entity_list_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(' ')

                    if len(parts) >= 3:
                        try:
                            remap_id = int(parts[1])
                            entity_name = parts[2]

                            entity_name = self.process_name(entity_name)

                            if entity_name:
                                entity_name_mapping[remap_id] = entity_name

                        except ValueError:
                            continue

                    elif len(parts) == 2:
                        try:
                            org_id = parts[0]
                            remap_id = int(parts[1])

                            entity_name = self.process_name(org_id)

                            if entity_name and not entity_name.isdigit():
                                entity_name_mapping[remap_id] = entity_name

                        except ValueError:
                            continue

        except FileNotFoundError:
            pass
        except Exception as e:
            pass

        return entity_name_mapping

    def extract_relation_name_from_url(self, url_string):
        if not url_string:
            return url_string

        is_inverse = False
        if url_string.startswith('inverse_'):
            is_inverse = True
            url_string = url_string[8:]

        separators = ['/', '#']

        relation_name = url_string
        for sep in separators:
            if sep in url_string:
                relation_name = url_string.split(sep)[-1]

        if not relation_name or relation_name == url_string:
            if 'ontology/' in url_string:
                relation_name = url_string.split('ontology/')[-1]
            elif 'property/' in url_string:
                relation_name = url_string.split('property/')[-1]
            elif 'rdf-syntax-ns#' in url_string:
                relation_name = url_string.split('rdf-syntax-ns#')[-1]
            elif 'rdfs#' in url_string:
                relation_name = url_string.split('rdfs#')[-1]
            elif 'gold/' in url_string:
                relation_name = url_string.split('gold/')[-1]

        relation_name = relation_name.strip()

        import re
        relation_name = re.sub('(?<!^)(?=[A-Z])', ' ', relation_name)

        if is_inverse:
            relation_name = f"inverse {relation_name}"

        return relation_name

    def load_relation_name_mapping(self, dataset_path):
        relation_name_mapping = {}
        relation_list_file = f'{dataset_path}/relation_list.txt'

        try:
            with open(relation_list_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(' ')
                    if len(parts) >= 2:
                        try:
                            relation_text = parts[0]
                            original_relation_id = int(parts[1])

                            clean_relation = self.process_name(relation_text)
                            clean_relation = clean_relation.replace('.', ' ')

                            clean_relation = self.extract_relation_name_from_url(clean_relation)

                            remapped_relation_id = original_relation_id + 2
                            relation_name_mapping[remapped_relation_id] = clean_relation

                            if self.loader.config.get('inverse_r', True):
                                inverse_relation_id = remapped_relation_id + 13
                                relation_name_mapping[inverse_relation_id] = f"inverse {clean_relation}"

                        except ValueError:
                            continue

        except FileNotFoundError:
            pass
        except Exception as e:
            pass

        return relation_name_mapping

    def load_interest_name_mapping(self, intent_path):
        interest_name_mapping = {}
        intent_list_file = f'{intent_path}/intent_list.txt'

        try:
            with open(intent_list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            original_interest_id = int(parts[0])
                            interest_name = parts[1]

                            interest_name_mapping[original_interest_id] = interest_name
                        except ValueError:
                            continue

        except FileNotFoundError:
            pass
        except Exception as e:
            pass

        return interest_name_mapping

    def build_node_mappings(self):
        self.node_names = {}
        user_num = self.loader.config['users']
        dataset_path = f'./data/{self.loader.config["dataset"]}'
        intent_path = f'./User_Intent/{self.loader.config["dataset"]}'

        item_name_mapping = self.load_item_name_mapping(dataset_path)
        entity_name_mapping = self.load_entity_name_mapping(dataset_path)
        relation_name_mapping = self.load_relation_name_mapping(dataset_path)
        interest_name_mapping = self.load_interest_name_mapping(intent_path)

        for uid in range(user_num):
            self.node_names[uid] = f"User_{uid}"

        for remapped_item_id in self.loader.train['itemid'].unique():
            original_item_id = remapped_item_id - user_num
            if original_item_id in item_name_mapping:
                item_name = item_name_mapping[original_item_id]
                if len(item_name) > 50:
                    item_name = item_name[:47] + "..."
                self.node_names[remapped_item_id] = f'"{item_name}"'
            else:
                self.node_names[remapped_item_id] = f"Item_{original_item_id}"

        for remapped_interest_id in self.loader.kg_interest['interest'].unique():
            original_interest_id = remapped_interest_id - user_num
            if original_interest_id in interest_name_mapping:
                interest_name = interest_name_mapping[original_interest_id]
                self.node_names[remapped_interest_id] = f'Interest: "{interest_name}"'
            else:
                self.node_names[remapped_interest_id] = f"Interest_{original_interest_id}"

        kg_entities = set(self.loader.kg_org['head'].unique()) | set(self.loader.kg_org['tail'].unique())
        for remapped_entity_id in kg_entities:
            if remapped_entity_id not in self.node_names:
                original_entity_id = remapped_entity_id - user_num
                if original_entity_id in entity_name_mapping:
                    entity_name = entity_name_mapping[original_entity_id]
                    if len(entity_name) > 50:
                        entity_name = entity_name[:47] + "..."
                    self.node_names[remapped_entity_id] = f'"{entity_name}"'
                else:
                    self.node_names[remapped_entity_id] = f"Entity_{original_entity_id}"

        self.relation_names = {}

        self.relation_names.update(relation_name_mapping)

        self.relation_names[self.loader.INTERACT_RELATION] = "interacted with"
        if hasattr(self.loader, 'INTERACT_RELATION_INV'):
            self.relation_names[self.loader.INTERACT_RELATION_INV] = "was interacted by"

        self.relation_names[self.loader.INTEREST_RELATION] = "has interest in"
        if hasattr(self.loader, 'INTEREST_RELATION_INV'):
            self.relation_names[self.loader.INTEREST_RELATION_INV] = "is interest of"

        for rel_id in range(self.loader.config['relations']):
            if rel_id not in self.relation_names:
                self.relation_names[rel_id] = f"relation_{rel_id}"

    def get_neighbors_with_relations(self, node_id):
        neighbors = []
        if node_id in self.loader.kg_dict:
            for rel_id, neighbor_id in self.loader.kg_dict[node_id]:
                neighbors.append((node_id, rel_id, neighbor_id))
        return neighbors

    def format_triplets_for_llm(self, triplets):
        formatted = []
        for head, rel, tail in triplets:
            head_name = self.node_names.get(head, f"Unknown_{head}")
            tail_name = self.node_names.get(tail, f"Unknown_{tail}")
            rel_name = self.relation_names.get(rel, f"relation_{rel}")
            if rel == self.loader.INTERACT_RELATION:
                formatted.append(f"{head_name} → {rel_name} → {tail_name}")
            elif rel == self.loader.INTEREST_RELATION:
                formatted.append(f"{head_name} → {rel_name} → {tail_name}")
            else:
                formatted.append(f"{head_name} --[{rel_name}]-> {tail_name}")

        return formatted

    def get_selection_prompt(self, center_node, triplets, node_type, top_k=5):
        center_name = self.node_names.get(center_node, f"Unknown_{center_node}")
        formatted_triplets = self.format_triplets_for_llm(triplets)

        if node_type == "user":
            task_description = "understand this user's preferences and interests to provide better recommendations"
            selection_criterion = "most informative for understanding user preferences"
            context_hint = """Consider both user's actual interactions (items they bought/rated/clicked) and their stated interests. 

    **IMPORTANT**: Actual user interactions provide concrete behavioral evidence of preferences, while interests show potential preferences. For a comprehensive understanding:
    - You MUST include at least 1 relationship showing actual user interaction with items (not just interests)
    - Balance between behavioral evidence (interactions) and preference indicators (interests)
    - Avoid selecting only interest-based relationships - mix them with interaction-based ones

    Prioritize relationships that show proven user behavior alongside their stated preferences."""

        else:
            task_description = "understand this item's characteristics and features to improve recommendation accuracy"
            selection_criterion = "most informative for understanding item features and characteristics"
            context_hint = "Consider relationships that reveal item attributes, categories, similar items, and descriptive features."

        prompt = f"""You are helping to build a knowledge subgraph for a recommendation system.

    **Current Focus**: {center_name} (a {node_type})
    **Goal**: Select relationships that help {task_description}

    **Available Relationships** ({len(formatted_triplets)} total):
    {chr(10).join([f"{i + 1:2d}. {trip}" for i, trip in enumerate(formatted_triplets)])}

    **Selection Criteria**: Choose the {top_k} relationships that are {selection_criterion}.

    **Guidance**: {context_hint}

    **Output Format**: Return only the numbers (1-{len(formatted_triplets)}) separated by commas.
    Example: "1,5,8,12,15"

    Your selection:"""

        return prompt


    def get_quality_assessment_prompt(self, sampled_subgraph, center_node, node_type):
        center_name = self.node_names.get(center_node, f"Unknown_{center_node}")
        formatted_triplets = self.format_triplets_for_llm(sampled_subgraph)

        if node_type == "user":
            assessment_question = f"Can we understand {center_name}'s preferences well enough for recommendations?"
            quality_criteria = [
                "User interest coverage (books, genres, topics they like)",
                "Behavioral patterns (interaction history, preferences)",
                "Preference diversity (different types of interests)"
            ]
        else:
            assessment_question = f"Do we have enough information about {center_name} for recommendations?"
            quality_criteria = [
                "Item characteristics (genre, author, topic, features)",
                "Item relationships (similar items, categories)",
                "Descriptive attributes (what makes this item unique)"
            ]

        criteria_text = "\n".join([f"- {criterion}" for criterion in quality_criteria])

        prompt = f"""Evaluate the quality of this knowledge subgraph for recommendation purposes.

    **Target**: {center_name} (a {node_type})
    **Current Subgraph** ({len(formatted_triplets)} relationships):
    {chr(10).join([f"• {trip}" for trip in formatted_triplets])}

    **Question**: {assessment_question}

    **Evaluation Criteria**:
    {criteria_text}

    **Response**: Only output "SUFFICIENT" or "INSUFFICIENT" - no explanation needed.

    Your assessment:"""
        return prompt

    def call_llm(self, prompt, max_tokens=500):
        url = "https://xh.v1api.cc/v1/chat/completions"

        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are an expert in graph sampling for recommendation systems."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }

        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': 'Bearer sk-NCe9VNEDxfEIZwG7DlJdLaGhvOW1Oyhuh7dWwMP7bbPUB5Dm'
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                response_json = response.json()

                if 'choices' in response_json and len(response_json['choices']) > 0:
                    content = response_json['choices'][0]['message']['content']
                    return content.strip()
                else:
                    return None
            else:
                return None

        except requests.exceptions.RequestException as e:
            return None
        except json.JSONDecodeError as e:
            return None
        except Exception as e:
            return None

    def parse_selection_response(self, response, total_count):
        try:
            numbers = []
            for part in response.split(','):
                num = int(part.strip())
                if 1 <= num <= total_count:
                    numbers.append(num - 1)
            return numbers
        except:
            return list(range(min(5, total_count)))

    def sample_subgraph(self, center_node, node_type, top_k=5):
        sampled_triplets = []
        current_frontier = [center_node]
        visited_nodes = {center_node}

        for hop in range(self.max_hops):
            next_frontier = []

            for node in current_frontier:
                neighbors = self.get_neighbors_with_relations(node)
                if not neighbors:
                    continue

                unvisited_neighbors = [
                    (h, r, t) for h, r, t in neighbors
                    if t not in visited_nodes
                ]

                if not unvisited_neighbors:
                    continue

                selection_prompt = self.get_selection_prompt(
                    node, unvisited_neighbors, node_type, top_k
                )

                llm_response = self.call_llm(selection_prompt)
                if not llm_response:
                    continue

                selected_indices = self.parse_selection_response(
                    llm_response, len(unvisited_neighbors)
                )

                for idx in selected_indices:
                    if idx < len(unvisited_neighbors):
                        triplet = unvisited_neighbors[idx]
                        sampled_triplets.append(triplet)
                        next_frontier.append(triplet[2])
                        visited_nodes.add(triplet[2])

            if sampled_triplets:
                quality_prompt = self.get_quality_assessment_prompt(
                    sampled_triplets, center_node, node_type
                )

                quality_response = self.call_llm(quality_prompt)

                if quality_response and quality_response.strip().upper() == "SUFFICIENT":
                    break

            current_frontier = list(set(next_frontier))
            if not current_frontier:
                break

            time.sleep(0.5)

        return sampled_triplets

    def batch_sample_subgraphs(self, node_list, node_type, top_k=5):
        results = {}

        for i, node_id in enumerate(node_list):
            subgraph = self.sample_subgraph(node_id, node_type, top_k)
            results[node_id] = subgraph

        return results

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_checkpoint(checkpoint_file, user_idx, item_idx, user_subgraphs, item_subgraphs):
    checkpoint_data = {
        'user_idx': user_idx,
        'item_idx': item_idx,
        'user_subgraphs': {str(k): v for k, v in user_subgraphs.items()},
        'item_subgraphs': {str(k): v for k, v in item_subgraphs.items()}
    }

    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2, cls=NumpyEncoder)
    return

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)

        user_subgraphs = {int(k): v for k, v in checkpoint_data['user_subgraphs'].items()}
        item_subgraphs = {int(k): v for k, v in checkpoint_data['item_subgraphs'].items()}

        return checkpoint_data['user_idx'], checkpoint_data['item_idx'], user_subgraphs, item_subgraphs
    else:
        return 0, 0, {}, {}

def main():
    args = parse_args()
    dataset = args.dataset
    config = json.load(open(f'./config/{dataset}.json'))
    config['device'] = f'cuda:{args.gpu_id}'
    seed = args.seed
    Ks = config['Ks']
    init_seed(seed, True)

    loader = MyLoader(config)
    graph = build_graph(loader, inverse=True)
    graph = graph.to(config['device'])

    sampler = LLMGraphSampler(loader)

    save_interval = 1

    save_dir = f'./subgraph sampling/{dataset}'
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_file = os.path.join(save_dir, f'checkpoint_{dataset}.json')
    final_result_file = os.path.join(save_dir, f'all_subgraphs_{dataset}.json')

    all_users = list(range(loader.config['users']))
    all_items = list(range(loader.config['items']))

    n = 5
    all_items = [item + len(all_users) for item in range(loader.config['items'])]

    user_start_idx, item_start_idx, user_subgraphs, item_subgraphs = load_checkpoint(checkpoint_file)

    try:
        for i in range(user_start_idx, len(all_users)):
            user_id = all_users[i]

            try:
                user_subgraph = sampler.batch_sample_subgraphs([user_id], "user", top_k=5)
                user_subgraphs.update(user_subgraph)

                if (i + 1) % save_interval == 0:
                    save_checkpoint(checkpoint_file, i + 1, item_start_idx, user_subgraphs, item_subgraphs)

            except Exception as e:
                save_checkpoint(checkpoint_file, i, item_start_idx, user_subgraphs, item_subgraphs)
                raise

        for i in range(item_start_idx, len(all_items)):
            item_id = all_items[i]

            try:
                item_subgraph = sampler.batch_sample_subgraphs([item_id], "item", top_k=3)
                item_subgraphs.update(item_subgraph)

                if (i + 1) % save_interval == 0:
                    save_checkpoint(checkpoint_file, len(all_users), i + 1, user_subgraphs, item_subgraphs)

            except Exception as e:
                save_checkpoint(checkpoint_file, len(all_users), i, user_subgraphs, item_subgraphs)
                raise

        final_results = {
            'user_subgraphs': {str(k): v for k, v in user_subgraphs.items()},
            'item_subgraphs': {str(k): v for k, v in item_subgraphs.items()},
            'summary': {
                'total_users': len(user_subgraphs),
                'total_items': len(item_subgraphs),
                'completion_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }

        with open(final_result_file, 'w') as f:
            json.dump(final_results, f, indent=2)

    except KeyboardInterrupt:
        current_user_idx = user_start_idx if 'i' not in locals() else i
        current_item_idx = item_start_idx
        save_checkpoint(checkpoint_file, current_user_idx, current_item_idx, user_subgraphs, item_subgraphs)

    except Exception as e:
        current_user_idx = user_start_idx if 'i' not in locals() else i
        current_item_idx = item_start_idx
        save_checkpoint(checkpoint_file, current_user_idx, current_item_idx, user_subgraphs, item_subgraphs)
        raise


if __name__ == "__main__":
    main()
