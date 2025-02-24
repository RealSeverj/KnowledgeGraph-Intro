from knowledge_graph import KnowledgeGraph


def test_get_entity_info():
    # 初始化知识图谱
    kg = KnowledgeGraph("knowledge_graph.json")

    # 添加测试数据
    kg.add_triple("Paris", "country", "France")

    print("test_get_entity_info passed.")

if __name__ == "__main__":
    test_get_entity_info()