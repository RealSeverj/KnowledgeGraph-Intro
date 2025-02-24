<h3>一个基础的知识图谱模型</h3>

基于DeepSeek-R1-Distill-Qwen-1.5B，保留对话历史与基础的知识三元组记录功能

安装依赖：
```pip install -r requirements.txt```

请确保网络通畅后下载模型：

```huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir DeepSeek-R1-Distill-Qwen-1.5B```

运行：
```python main.py```

prompt与生成相关参数请参考model_intergration.py与config.py