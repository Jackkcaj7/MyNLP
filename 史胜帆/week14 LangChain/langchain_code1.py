from dotenv import find_dotenv,load_dotenv
import os
from langchain_openai import ChatOpenAI



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
    #大语言模型调用
    response = model.invoke("用粤语跟我答个热情招呼")
    print(response.content)