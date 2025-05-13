from langchain_ollama import OllamaLLM  # 修改导入路径
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 连接到Ollama服务
llm = OllamaLLM(
    base_url="http://localhost:11434",  # Ollama默认端口
    model="qwen3:1.7b"  # 替换为你部署的千问模型名称
)

# 创建简单的问答链
template = """
问题：{question}
回答："""

prompt = PromptTemplate(template=template, input_variables=["question"])

# 使用新的Runnable接口
# 修改Runnable接口的使用方式
qa_chain = (
    {"question": RunnablePassthrough()}
    | prompt
    | (lambda x: llm.invoke(x.to_string()))  # 修改为使用to_string()
)

# 提问
question = "什么是小六壬?"
answer = qa_chain.invoke(question)  # 直接传入question
print(f"问题：{question}\n回答：{answer}")

# 删除重复的调用
# answer = qa_chain.invoke(question)print(f"问题：{question}\n回答：{answer}")
