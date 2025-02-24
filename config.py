class Config:
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    KG_FILE = "knowledge_graph.json"
    MAX_DIALOGUE_HISTORY = 3  # 保留最近3轮对话
    DEVICE = "cpu"  # 改为"cpu"若无GPU
