"""
通义千问辅助模块 - 对话式AI助手
"""

import dashscope
from dashscope import Generation
from typing import Dict, List
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QianWenHelper:
    def __init__(self):
        self.model = 'qwen-turbo'
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            self.api_key = "sk-85ce554faa624e3aa748427e19839da3"
            os.environ['DASHSCOPE_API_KEY'] = self.api_key

        dashscope.api_key = self.api_key
        logger.info(f"AI助手初始化完成，模型: {self.model}")

        self.system_prompt = """你是校园失物招领AI助手。

回复要简洁实用：
1. 直接回答问题
2. 重点突出
3. 必要时给链接

页面链接（请用完整URL）：
- 提交丢失：http://localhost:5000/submit/lost
- 提交招领：http://localhost:5000/submit/found  
- 查看丢失：http://localhost:5000/lost
- 查看招领：http://localhost:5000/found
- 匹配：http://localhost:5000/match

示例：
用户：校园卡丢了？
AI：去食堂图书馆找找。没有就<a href="/submit/lost">提交丢失信息</a>。

用户：怎么写招领启事？
AI：【招领启事】
拾获：校园卡
时间：今天下午
地点：图书馆
特征：蓝色卡套
联系：李同学 138****5678
<a href="/submit/found">点此提交</a>

现在开始，简洁实用地帮助用户。"""

    def chat(self, user_message: str, conversation_history: List[Dict] = None) -> str:
        try:
            logger.info(f"用户消息: {user_message[:50]}...")

            messages = [{'role': 'system', 'content': self.system_prompt}]

            if conversation_history:
                messages.extend(conversation_history[-6:])

            messages.append({'role': 'user', 'content': user_message})

            response = Generation.call(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )

            if response.status_code == 200:
                if hasattr(response, 'output') and response.output:
                    if hasattr(response.output, 'text') and response.output.text:
                        ai_reply = response.output.text
                    elif hasattr(response.output, 'choices') and response.output.choices:
                        if len(response.output.choices) > 0:
                            choice = response.output.choices[0]
                            if hasattr(choice, 'message') and choice.message:
                                if hasattr(choice.message, 'content'):
                                    ai_reply = choice.message.content
                                else:
                                    ai_reply = "回复格式错误。"
                            else:
                                ai_reply = "回复异常。"
                        else:
                            ai_reply = "无有效回复。"
                    else:
                        ai_reply = "回复结构异常。"
                else:
                    ai_reply = "服务响应异常。"

                logger.info(f"AI回复长度: {len(ai_reply)} 字符")
                return ai_reply

            else:
                logger.error(f"API调用失败: {response.code}")
                return "网络连接异常，暂时无法使用AI服务。"

        except Exception as e:
            logger.error(f"对话处理异常: {str(e)}")
            return self._get_fallback_response(user_message, conversation_history)

    def _get_fallback_response(self, user_message: str, conversation_history: List[Dict] = None) -> str:
        msg_lower = user_message.lower()

        if any(word in msg_lower for word in ['啥意思', '什么意思', '为什么', '怎么解决', '怎么办']):
            if conversation_history:
                for msg in reversed(conversation_history):
                    if msg['role'] == 'assistant':
                        last_ai_reply = msg['content']
                        if any(err in last_ai_reply for err in ['不可用', '连接异常', '网络连接', '服务暂时']):
                            return """网络连接异常。

可能原因：
1. 网络不稳定
2. API服务故障
3. 本地网络问题

您可以：
1. 检查网络
2. <a href="/submit/lost">直接提交丢失</a>
3. <a href="/found">查看招领物品</a>
4. 稍后再试"""

        return self._get_offline_response(msg_lower)

    def _get_offline_response(self, user_message: str) -> str:
        if any(word in user_message for word in ['校园卡', '学生卡', '饭卡']):
            if '丢' in user_message:
                return """校园卡丢失：
1. 立即挂失
2. 检查食堂图书馆
3. <a href="/submit/lost">提交丢失信息</a>
4. <a href="/found">查看招领</a>"""

        elif '手机' in user_message:
            return """手机丢失：
1. 挂失SIM卡
2. 使用查找功能
3. <a href="/submit/lost">提交信息</a>"""

        elif '描述' in user_message:
            return """物品描述要点：
• 名称
• 颜色大小
• 品牌特征
• 时间地点

告诉我具体物品。"""

        elif any(word in user_message for word in ['启事', '模板', '怎么写']):
            return """启事模板：
【寻物/招领启事】
物品：[名称]
时间：[时间]
地点：[位置]
特征：[描述]
联系：[方式]

<a href="/submit/lost">直接提交</a>"""

        else:
            return """网络连接异常。

替代方案：
• <a href="/submit/lost">提交丢失</a>
• <a href="/submit/found">提交招领</a>
• <a href="/found">查看招领</a>
• <a href="/lost">查看丢失</a>

稍后再试。"""

    def get_welcome_message(self) -> str:
        return """校园失物招领助手。

需要：描述物品、写启事、答疑？

直接说需求。"""


qianwen = QianWenHelper()