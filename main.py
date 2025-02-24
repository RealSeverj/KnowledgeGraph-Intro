from knowledge_graph import KnowledgeGraph
from dialogue_memory import DialogueMemory
from model_integration import DeepSeekKGModel
from config import Config
import re


def extract_entities(text):
    """简单实体提取（可根据需求替换为NER模型）"""
    return list(set(re.findall(r'\b[A-Z][a-z]+\b', text)))


def main():
    # 初始化组件
    config = Config()
    kg = KnowledgeGraph(config.KG_FILE)
    memory = DialogueMemory(config.MAX_DIALOGUE_HISTORY)
    model = DeepSeekKGModel()

    kg.add_triple("Severj", "is the master of", "Shire")
    # Who is Severj?

    print("System start（Press 'exit' to exit）")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            kg.save()
            break

        # 知识图谱检索
        entities = extract_entities(user_input)
        print(f"Entities Detected：{entities}")
        kg_text = ""
        for entity in entities:
            info = kg.get_entity_info(entity)
            if info:
                kg_text += f"{entity} {info}. "
                print(f'Knowledege Graph Return: {kg_text}')

        print('Generating response...\n')

        # 构建提示
        history_text = memory.get_formatted_history()
        prompt = user_input
        kg_text = kg_text if kg_text else "No relevant knowledge"
        history_text = history_text if history_text else "No history"

        # 生成回复
        response = model.deepseek_infer(prompt, history_text, kg_text)

        def process_response(raw_response):
            """提取英文响应内容"""
            reasoning = ""
            answer = raw_response
            if "</think>" in raw_response:
                # 处理英文标签
                reasoning_part, answer_part = raw_response.split("</think>")
                # 清理多余换行
                reasoning = "\n".join([line.strip() for line in reasoning_part.split("\n") if line.strip()])
                answer = " ".join(answer_part.replace("\n", " ").split())
            return answer, reasoning

        final_answer, reasoning = process_response(response)
        print(f"Think：\n{reasoning}\n")
        print(f"Answer: {final_answer}")

        # 更新记忆
        memory.add_entry(user_input, response)


if __name__ == "__main__":
    main()
