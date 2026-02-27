from langchain_core.prompts import ChatPromptTemplate
#用类似fstr格式占位符的形式格式化调用模型
from dotenv import find_dotenv,load_dotenv
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser #大模型输出解析器


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

    #创建大语言模型输出结果解析器对象
    parser = StrOutputParser()

    #创建template对象
    prompt_temp = ChatPromptTemplate.from_messages(
        [('system','你是精通{language}的AI助理，可以将普通话翻译为{language}'),
         ('user','{text}')
         ]
    )

    # #调用template对象
    # messages = prompt_temp.invoke({"language":"四川话","text":"早上好 四川话怎莫说"})
    # response = model.invoke(messages)
    #调用解析器
    #content = parser.invoke(response)

    #上面2次调用invoke的方式没问题但有些繁琐
    #优化 采用langchain表达式语言(LCEL)
    #原理是采用Linux系统中通道运算符实现 把上一次运算的结果直接作为下一次运算的输入
    #通道运算符 | 
    chain = prompt_temp | model | parser #构建1条链条 此处langchain的意义开始显现
    content = chain.invoke({"language":"北京话","text":"早上好 怎莫说"})

    print(content)

    #可以理解调用大模型进行语言翻译就是这样实现的