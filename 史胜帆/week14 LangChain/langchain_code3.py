from langchain_core.prompts import ChatPromptTemplate
#用类似fstr格式占位符的形式格式化调用模型

if __name__ == "__main__":
    #创建template对象
    prompt_temp = ChatPromptTemplate.from_messages(
        [('system','你是精通{language}的AI助理，可以将普通话翻译为{language}'),
         ('user','{text}')
         ]
    )

    #调用template对象
    res = prompt_temp.invoke({"language":"四川话","text":"早上好 四川话怎莫说"})