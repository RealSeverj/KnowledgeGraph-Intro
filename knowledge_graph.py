import json
from collections import defaultdict


class KnowledgeGraph:
    def __init__(self, file_path):
        self.file_path = file_path
        self.graph = defaultdict(dict)
        self.load()

    def add_triple(self, entity, relation, value):
        self.graph[entity][relation] = value
        self.save()

    def get_entity_info(self, entity):
        return self.graph.get(entity, {})

    def save(self):
        with open(self.file_path, 'w') as f:
            json.dump(dict(self.graph), f)

    def load(self):
        try:
            with open(self.file_path, 'r') as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    self.graph = defaultdict(dict, data)
                else:
                    self.graph = defaultdict(dict)
        except FileNotFoundError:
            self.graph = defaultdict(dict)
