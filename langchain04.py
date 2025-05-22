from langchain.prompts import PromptTemplate, FewShotPromptTemplate, PipelinePromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

# 创建一个反义词的任务示例
examples = [
    {"input":"good","output":"bad"},
    {"input":"happy","output":"sad"},
    {"input":"rich","output":"poor"},
    {"input":"beautiful","output":"ugly"},
    {"input":"smart","output":"stupid"},
    {"input":"fast","output":"slow"},
    {"input":"tall","output":"short"},
]

example_prompt = PromptTemplate(
    input_variables=["input","output"],
    template="输入：{input}\n输出：{output}"
)

example_selector = LengthBasedExampleSelector(
    examples = examples,
    example_prompt = example_prompt,
    max_length = 20
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt = example_prompt,
    prefix = "请将以下单词转换为反义词。",
    suffix = "输入：{input}\n输出：",
    input_variables = ["input"]
)

# 定义一个指令解释提示模板
instruction_prompt = PromptTemplate(
    input_variables=["dynamic_prompt"],
    template="以下是一个反义词转换任务的提示，请按照提示完成任务。\n{dynamic_prompt}"
)

# 创建管道提示模板
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=instruction_prompt,
    pipeline_prompts=[
        ("dynamic_prompt", dynamic_prompt)
    ]
)

# 使用管道提示模板生成最终提示并打印
print(pipeline_prompt.format(input="happy"))
print(dynamic_prompt.format(input="good"))