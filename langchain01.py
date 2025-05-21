from langchain_ollama import OllamaLLM  # 修改导入路径
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from fastapi import FastAPI
from pydantic import BaseModel  # 导入 BaseModel

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

app = FastAPI(title="Ollama API", version="1.0", description="Ollama API")

# 添加根路由，返回欢迎信息
@app.get("/")
def read_root():
    return {"message": "Welcome to the Ollama API!"}

# 定义请求体模型
class QuestionRequest(BaseModel):
    question: str

# 添加问答路由，使用 POST 方法接收问题
@app.post("/qa")
async def ask_question(request: QuestionRequest):
    result = qa_chain.invoke({"question": request.question})
    return {"answer": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
