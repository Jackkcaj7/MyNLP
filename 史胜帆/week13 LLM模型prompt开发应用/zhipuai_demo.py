import os
from dotenv import load_dotenv,find_dotenv
from openai import OpenAI

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    #创建调用API的客户端
    client = OpenAI(
        api_key=os.environ['api_key'],
        base_url=os.environ['base_url']
    )

    #chat模式的模型调用
    response = client.chat.completions.create(
        #模型名称
        model = "glm-z1-flash",
        #消息列表(提示词prompt) [jsons] 也可以理解为消息记录
        messages=[
            {'role':'user','content':'你吃饭了吗？'}
        ], #role是角色 一般user content是发起对话的内容 普通交流
        #role 还可以是system 可以用于AI角色设定 content告诉它是一个具体人物 系统不会根据content输出内容
        #assistant content可以写我们想让AI知道的内容 作为AI后续生成对话的条件 AI不会做出输出 用于AI根据这些上下文context信息更像人交流
        #tool 让AI读取content指令调用特定程序 比如content可以是打开空调
        # 因此role是和大模型交流之前对它的角色设定
        #  
        #模型参数
        temperature=0,#采样温度 取值[0,1] 越小 模型输出结果随机越小 每个用户得到的结果不同
        #max_tokens 一定程度限制模型输出冗余信息
        # 包括input(context)和max output
        #context是每次对话模型的输入的最大token 即每次对话之前参考的前文信息
        #max ouput 是每次对话的输出最大token数量
    )

    #打印结果
    print(response.choices[0].message.content) #调用API相应内容中的content输出
