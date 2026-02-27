from dotenv import find_dotenv,load_dotenv
import os
from langchain_openai import ChatOpenAI #LLM调用封装
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage 
#相当于对话角色role user assistant system


if __name__ == "__main__":
    #加载env文件配置项
    load_dotenv(find_dotenv())
    #初始化创建大语言模型
    model = ChatOpenAI(
        model = "GLM-4-Flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.75
    )

    #带角色定制的消息
    messages = [
        SystemMessage(content="你是一个精通粤语的AI助手，还能做普通话粤语双向翻译"),
        HumanMessage(content="你好 早上好 粤语怎么说"),

    ]

    #基于messages列表的调用
    response = model.invoke(messages)
    print(response.content)

    #基于key:value字典的模型调用 有时候版本不支持
    response = model.invoke({"messages":messages})
    print(response.content)

#prompttemplates提示模板 把原始用户输入转换为模型输入列表