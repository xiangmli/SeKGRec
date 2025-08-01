import json
import os
from collections import defaultdict


class SubgraphToText:
    def __init__(self, dataset_name, user_num=5576):
        self.dataset_name = dataset_name
        self.user_num = user_num
        self.dataset_path = f'./data/{dataset_name}'
        self.intent_path = f'./User_Intent/{dataset_name}'

        self.entity_name_mapping = self.load_entity_name_mapping()
        self.relation_name_mapping = self.load_relation_name_mapping()
        self.interest_name_mapping = self.load_interest_name_mapping()

        self.INTERACT_RELATION = 0
        self.INTERACT_RELATION_INV = 1
        self.INTEREST_RELATION = 28
        self.INTEREST_RELATION_INV = 29

        self.relation_name_mapping[self.INTERACT_RELATION] = "interacted with"
        self.relation_name_mapping[self.INTERACT_RELATION_INV] = "was interacted by"
        self.relation_name_mapping[self.INTEREST_RELATION] = "has interest in"
        self.relation_name_mapping[self.INTEREST_RELATION_INV] = "is interest of"

    def process_name(self, name):
        """Process dataset-specific name formats"""
        if name.startswith('http://dbpedia.org/resource/'):
            name = name.replace('http://dbpedia.org/resource/', '')
            name = name.replace('_', ' ')
        return name

    def extract_relation_name_from_url(self, url_string):
        """Extract key parts from URL-formatted relation names"""
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

    def load_entity_name_mapping(self):
        """Load entity name mapping"""
        entity_name_mapping = {}
        entity_list_file = f'{self.dataset_path}/entity_list.txt'

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

            print(f"Successfully loaded {len(entity_name_mapping)} entity names")
        except FileNotFoundError:
            print(f"Warning: Entity list file not found {entity_list_file}")
        except Exception as e:
            print(f"Error reading entity list: {e}")

        return entity_name_mapping

    def load_relation_name_mapping(self):
        """Load relation name mapping"""
        relation_name_mapping = {}
        relation_list_file = f'{self.dataset_path}/relation_list.txt'

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

                            inverse_relation_id = remapped_relation_id + 13
                            relation_name_mapping[inverse_relation_id] = f"inverse {clean_relation}"

                        except ValueError:
                            continue

            print(f"Successfully loaded {len(relation_name_mapping)} relation names")
        except FileNotFoundError:
            print(f"Warning: Relation list file not found {relation_list_file}")
        except Exception as e:
            print(f"Error reading relation list: {e}")

        return relation_name_mapping

    def load_interest_name_mapping(self):
        """Load interest name mapping"""
        interest_name_mapping = {}
        intent_list_file = f'{self.intent_path}/intent_list.txt'

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

            print(f"Successfully loaded {len(interest_name_mapping)} interest names")
        except FileNotFoundError:
            print(f"Warning: Interest list file not found {intent_list_file}")
        except Exception as e:
            print(f"Error reading interest list: {e}")

        return interest_name_mapping

    def get_node_name(self, node_id):
        """Get node name"""
        if node_id < self.user_num:
            return f"User_{node_id}"

        original_id = node_id - self.user_num

        if original_id in self.entity_name_mapping:
            return self.entity_name_mapping[original_id]

        if original_id in self.interest_name_mapping:
            return self.interest_name_mapping[original_id]

        return f"Entity_{original_id}"

    def get_relation_name(self, relation_id):
        """Get relation name"""
        if relation_id in self.relation_name_mapping:
            return self.relation_name_mapping[relation_id]
        return f"relation_{relation_id}"

    def _parse_to_rct_format(self, subgraph, center_node_id, center_name, node_type):
        """Directly convert triples to tree structure"""
        tree_structure = {
            'root': {
                'name': f"{node_type.title()} Profile Analysis",
                'center_node': {
                    'name': center_name,
                    'relations': {}
                }
            }
        }

        center_direct_relations = {}
        entity_to_relations = {}

        for triple in subgraph:
            head, relation, tail = triple
            head_name = self.get_node_name(head)
            tail_name = self.get_node_name(tail)
            relation_name = self.get_relation_name(relation)

            if head not in entity_to_relations:
                entity_to_relations[head] = []
            if tail not in entity_to_relations:
                entity_to_relations[tail] = []

            entity_to_relations[head].append((relation, tail, tail_name, 'outgoing'))
            entity_to_relations[tail].append((relation, head, head_name, 'incoming'))

            if head == center_node_id:
                if relation_name not in center_direct_relations:
                    center_direct_relations[relation_name] = []
                center_direct_relations[relation_name].append({
                    'entity_id': tail,
                    'entity_name': tail_name,
                    'direction': 'outgoing'
                })
            elif tail == center_node_id:
                inverse_relation = f"inverse_of_{relation_name}"
                if inverse_relation not in center_direct_relations:
                    center_direct_relations[inverse_relation] = []
                center_direct_relations[inverse_relation].append({
                    'entity_id': head,
                    'entity_name': head_name,
                    'direction': 'incoming'
                })

        tree_structure['root']['center_node']['relations'] = center_direct_relations

        for relation_name, entities in center_direct_relations.items():
            for entity_info in entities:
                entity_id = entity_info['entity_id']
                entity_name = entity_info['entity_name']

                entity_sub_relations = []
                if entity_id in entity_to_relations:
                    for rel, other_entity_id, other_entity_name, direction in entity_to_relations[entity_id]:
                        if other_entity_id != center_node_id:
                            rel_name = self.get_relation_name(rel)
                            entity_sub_relations.append({
                                'relation': rel_name,
                                'target': other_entity_name,
                                'target_id': other_entity_id,
                                'direction': direction
                            })

                entity_info['sub_relations'] = entity_sub_relations

        return self._generate_direct_tree_text(tree_structure)
    # tree structure for human recognition
    # def _generate_direct_tree_text(self, tree_structure):
    #     """Generate direct mapping tree text"""
    #     lines = []
    #
    #     root_info = tree_structure['root']
    #     center_info = root_info['center_node']
    #
    #     lines.append(f" {root_info['name']}")
    #     lines.append(f"├──  {center_info['name']}")
    #
    #     relations = list(center_info['relations'].items())
    #
    #     for i, (relation_name, entities) in enumerate(relations):
    #         is_last_relation = (i == len(relations) - 1)
    #         relation_prefix = "│   └── " if is_last_relation else "│   ├── "
    #
    #         display_relation = relation_name.replace("inverse_of_", "")
    #         lines.append(f"{relation_prefix} {display_relation}")
    #
    #         for j, entity_info in enumerate(entities):
    #             is_last_entity = (j == len(entities) - 1)
    #             has_sub_relations = bool(entity_info['sub_relations'])
    #
    #             if is_last_relation:
    #                 if is_last_entity and not has_sub_relations:
    #                     entity_prefix = "    └── "
    #                 else:
    #                     entity_prefix = "    ├── "
    #             else:
    #                 if is_last_entity and not has_sub_relations:
    #                     entity_prefix = "│   │   └── "
    #                 else:
    #                     entity_prefix = "│   │   ├── "
    #
    #             lines.append(f"{entity_prefix} {entity_info['entity_name']}")
    #
    #             if has_sub_relations:
    #                 sub_relations = entity_info['sub_relations']
    #                 for k, sub_rel in enumerate(sub_relations):
    #                     is_last_sub = (k == len(sub_relations) - 1)
    #
    #                     if is_last_relation:
    #                         if is_last_entity and is_last_sub:
    #                             sub_prefix = "        └── "
    #                         else:
    #                             sub_prefix = "        ├── "
    #                     else:
    #                         if is_last_entity and is_last_sub:
    #                             sub_prefix = "│   │       └── "
    #                         else:
    #                             sub_prefix = "│   │       ├── "
    #
    #                     if sub_rel['direction'] == 'outgoing':
    #                         relation_display = f"{sub_rel['relation']} → {sub_rel['target']}"
    #                     else:
    #                         relation_display = f"{sub_rel['target']} → {sub_rel['relation']}"
    #
    #                     lines.append(f"{sub_prefix} {relation_display}")
    #
    #     return "\n".join(lines)

    def _generate_direct_tree_text(self, tree_structure):
        """Generate direct mapping tree text"""
        lines = []

        root_info = tree_structure['root']
        center_info = root_info['center_node']

        lines.append(f"{root_info['name']}")
        lines.append(f"  {center_info['name']}")

        relations = list(center_info['relations'].items())

        for i, (relation_name, entities) in enumerate(relations):
            display_relation = relation_name.replace("inverse_of_", "")
            lines.append(f"    {display_relation}")

            for j, entity_info in enumerate(entities):
                has_sub_relations = bool(entity_info['sub_relations'])
                lines.append(f"      {entity_info['entity_name']}")

                if has_sub_relations:
                    sub_relations = entity_info['sub_relations']
                    for k, sub_rel in enumerate(sub_relations):
                        if sub_rel['direction'] == 'outgoing':
                            relation_display = f"{sub_rel['relation']} → {sub_rel['target']}"
                        else:
                            relation_display = f"{sub_rel['target']} → {sub_rel['relation']}"

                        lines.append(f"        {relation_display}")

        return "\n".join(lines)

    def _parse_to_traditional_format(self, subgraph, center_node_id, center_name):
        """Traditional format parsing (original logic)"""
        first_order_info = []
        higher_order_groups = defaultdict(lambda: defaultdict(list))

        for triple in subgraph:
            head, relation, tail = triple
            head_name = self.get_node_name(head)
            tail_name = self.get_node_name(tail)
            relation_name = self.get_relation_name(relation)

            if head == center_node_id or tail == center_node_id:
                first_order_info.append(f"{head_name} {relation_name} {tail_name}")
            else:
                if "interest" in relation_name.lower():
                    interest_entity = head_name
                    user_entity = tail_name
                    higher_order_groups[relation_name][interest_entity].append(user_entity)
                else:
                    higher_order_groups[relation_name]["general"].append(f"{head_name} and {tail_name}")

        first_order_text = ""
        if first_order_info:
            first_order_text = "First-order information:\n" + "\n".join([f"- {info}" for info in first_order_info])

        higher_order_text = ""
        if higher_order_groups:
            higher_order_lines = []

            for relation, entity_groups in higher_order_groups.items():
                if "interest" in relation.lower():
                    for interest_entity, users in entity_groups.items():
                        if users:
                            users_str = "、".join(users)
                            object_type = "users" if any("User" in user for user in users) else "books"
                            higher_order_lines.append(
                                f"The {object_type} with the same {relation} {interest_entity} also include: {users_str}"
                            )
                else:
                    for _, items in entity_groups.items():
                        if items:
                            items_str = "、".join(items)
                            object_type = "users" if any("User" in item for item in items) else "books"
                            higher_order_lines.append(
                                f"The {object_type} with the same {relation} also include: {items_str}"
                            )

            if higher_order_lines:
                higher_order_text = "Higher-order information:\n" + "\n".join(higher_order_lines)

        return first_order_text, higher_order_text

    def parse_subgraph_to_text(self, subgraph, center_node_id, node_type="user", use_rct=True):
        """Parse subgraph into natural language text

        Args:
            subgraph: list of triples in the subgraph
            center_node_id: center node ID
            node_type: node type ("user" or "item")
            use_rct: whether to use RCT format
        """
        center_name = self.get_node_name(center_node_id)

        if use_rct:
            return self._parse_to_rct_format(subgraph, center_node_id, center_name, node_type)
        else:
            return self._parse_to_traditional_format(subgraph, center_node_id, center_name)

    def generate_system_instruction(self, node_type, dataset_name):
        """Generate system instruction"""
        if node_type == "user":
            if dataset_name == "ml-1m":
                return (
                    "Assume you are an expert in movie recommendation. You will be given a user's interaction "
                    "history and interest information in natural language, along with some higher-order relationships. "
                    "Please analyze and summarize this user's movie preferences and characteristics. Your response "
                    "should be a coherent paragraph and no more than 150 words."
                )
            elif dataset_name in ("book-crossing", "dbbook2014"):
                return (
                    "Assume you are an expert in book recommendation. You will be given a user's reading history "
                    "and interest information in natural language, along with some higher-order relationships. "
                    "Please analyze and summarize this user's book preferences and reading characteristics. "
                    "Your response should be a coherent paragraph and no more than 150 words."
                )
        else:
            if dataset_name == "ml-1m":
                return (
                    "Assume you are an expert in movie recommendation. You will be given a movie's information in "
                    "natural language, along with some higher-order relationships. Please summarize the movie characteristics "
                    "and analyze what kind of users would like it. Your response should be a coherent paragraph and no more "
                    "than 200 words."
                )
            elif dataset_name in ("book-crossing", "dbbook2014"):
                return (
                    "Assume you are an expert in book recommendation. You will be given a book's information in "
                    "natural language, along with some higher-order relationships. Please summarize the book characteristics "
                    "and analyze what kind of users would like reading it. Your response should be a coherent paragraph and "
                    "no more than 200 words."
                )

        return "Please analyze the given information and provide insights."

    def process_subgraphs_to_jsonl(self, subgraph_file, output_file, subgraph_type="user_subgraphs"):
        """Process subgraphs and save as JSONL"""
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
        with open(subgraph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        subgraphs = data[subgraph_type]
        node_type = "user" if "user" in subgraph_type else "item"
        system_instruction = self.generate_system_instruction(node_type, self.dataset_name)
        results = []

        for center_id, subgraph in subgraphs.items():
            center_node_id = int(center_id)
            center_name = self.get_node_name(center_node_id)

            use_tree = True
            user_content = f"Target {node_type}: {center_name}\n\n"

            if use_tree:
                tree_content = self.parse_subgraph_to_text(subgraph, center_node_id, node_type, use_tree)
                user_content += tree_content
            else:
                first_order_text, second_order_text = self.parse_subgraph_to_text(
                    subgraph, center_node_id, node_type, use_tree
                )
                if first_order_text:
                    user_content += first_order_text + "\n\n"
                if second_order_text:
                    user_content += second_order_text

            entry = {
                "center_id": center_id,
                "center_name": center_name,
                "node_type": node_type,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ]
            }
            results.append(entry)

        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"Successfully processed {len(results)} {node_type} subgraphs and saved to {output_file}")
        return results


def main():
    dataset_name = "dbbook2014"
    user_num = 5576

    subgraph_file = f"./subgraph sampling/{dataset_name}/subgraphs.json"
    user_output_file = f"./batch_input/{dataset_name}/user_subgraph_llm_input.jsonl"
    item_output_file = f"./batch_input/{dataset_name}/item_subgraph_llm_input.jsonl"

    processor = SubgraphToText(dataset_name, user_num)

    if os.path.exists(subgraph_file):
        processor.process_subgraphs_to_jsonl(
            subgraph_file,
            user_output_file,
            "user_subgraphs"
        )
        processor.process_subgraphs_to_jsonl(
            subgraph_file,
            item_output_file,
            "item_subgraphs"
        )
    else:
        print(f"Subgraph file not found: {subgraph_file}")


if __name__ == "__main__":
    main()
