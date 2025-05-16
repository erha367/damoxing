from langchain_ollama import OllamaLLM  # 修改导入路径
from langchain_core.prompts import  ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# 连接到Ollama服务
llm = OllamaLLM(
    base_url="http://localhost:11434",  # Ollama默认端口
    model="qwen3:1.7b"  # 替换为你部署的千问模型名称
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的程序员，擅长使用各种编程语言编写代码。请根据对话历史和最新提问，给出答案。"),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

chain = prompt | llm
# 初始化消息历史记录
message_history = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="history",
)
# 模拟对话，使用相同的session_id
session_id = "user1"
# 与用户交互
print(chain_with_history.invoke({"input": "你好，你是谁？"}, config={"configurable": {"session_id": session_id}}))
print(chain_with_history.invoke({"input": "能帮我写个python 打印当前时间的程序么？"}, config={"configurable": {"session_id": session_id}}))