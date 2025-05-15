from langchain.chains.llm import LLMChain
from langchain_ollama import OllamaLLM  # 修改导入路径
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import  ChatPromptTemplate

# 连接到Ollama服务
llm = OllamaLLM(
    base_url="http://localhost:11434",  # Ollama默认端口
    model="qwen3:1.7b"  # 替换为你部署的千问模型名称
)

prompt = ChatPromptTemplate.from_template("""
    你是一位专业的程序员，擅长使用各种编程语言编写代码。请根据对话历史和最新提问，给出答案。
    对话历史：{history}
    最新提问：{input}
    回答：
""")

# 初始化内存
memory = ConversationBufferMemory(memory_key="history",return_messages=True)

# 使用新的Runnable接口，遵循LangChain新版内存使用方式
chain = LLMChain(
    llm = llm,
    prompt = prompt,
    memory = memory,
)

# 与用户交互
print(chain.invoke({"input": "你好，你是谁？"})["text"])
print(chain.invoke({"input": "能帮我写个python 打印当前时间的程序么？"})["text"])
print(chain.invoke({"input": "再帮我写一个c语言版本的吧？"})["text"])
