import os
from openai import OpenAI
from dotenv import load_dotenv

# 在脚本运行前加载.env文件
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "我是您的助手，可以帮助您回答问题，提供信息和支持。"
        },
        {
            "role": "user",
            "content": "你是谁?"
        },
    ],
    #  模型生成文本的随机性。温度值越低（接近0），模型生成的文本会更加保守和确定；温度值越高（接近1或更大），模型生成的文本会更加随机和创造性。
    temperature=1,
    # 模型生成文本的最大长度，以令牌（token）为单位。
    max_tokens=256,
    # 用于减少模型重复使用相同令牌的可能性。值越高，模型越不会重复相同的行话或短语。
    frequency_penalty=0,
    # 用于增加模型生成新话题和信息的可能性。值越高，模型越倾向于探讨之前未出现过的主题。这个参数可以帮助避免模型总是停留在已经讨论过的内容上。
    presence_penalty=0
)
print(response)
print(response.choices[0].message.content)