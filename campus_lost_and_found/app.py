"""
校园失物招领AI匹配平台 - BERT深度学习模型 + 通义千问API
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import json
import os
import datetime
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import traceback
from dateutil import parser  # 【新增】用于解析时间

# 导入通义千问辅助模块
from qianwen_helper import qianwen

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# 配置
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 确保目录存在
for dir_name in [UPLOAD_FOLDER, 'data', 'exports']:
    Path(dir_name).mkdir(exist_ok=True)

# 模拟数据库
lost_items = []
found_items = []
matches = []


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==================== BERT深度学习模型类 ====================
class BertMatcher:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"BERT模型 - 使用设备: {self.device}")
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.init_model()

    def init_model(self):
        try:
            print("正在加载BERT深度学习模型...")
            model_path = r"C:\Users\ASUS\Desktop\pythonwork\campus_lost_and_found\model"

            if os.path.exists(model_path):
                try:
                    print("加载本地BERT模型...")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                    self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
                    self.model = self.model.to(self.device)
                    self.model.eval()

                    # 测试模型
                    test_text = "测试文本"
                    inputs = self.tokenizer(test_text, return_tensors="pt", truncation=True, max_length=10)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.model(**inputs)

                    if outputs.last_hidden_state is not None:
                        print("BERT模型加载成功")
                        self.model_loaded = True
                    else:
                        self.load_online_model()

                except Exception as e:
                    print(f"本地BERT模型加载失败: {e}")
                    self.load_online_model()
            else:
                self.load_online_model()

        except Exception as e:
            print(f"BERT模型初始化错误: {e}")
            self.model_loaded = False

    def load_online_model(self):
        try:
            model_name = "bert-base-chinese"
            print(f"下载在线模型: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print("在线BERT模型加载成功")
        except Exception as e:
            print(f"在线模型加载失败: {e}")
            self.model_loaded = False

    def calculate_similarity(self, text1: str, text2: str) -> float:
        if not self.model_loaded:
            print("⚠️ BERT模型未加载，使用简单相似度")
            return float(self.simple_similarity(text1, text2))

        try:
            # 提取特征
            inputs1 = self.tokenizer(text1, return_tensors="pt", truncation=True, max_length=128)
            inputs2 = self.tokenizer(text2, return_tensors="pt", truncation=True, max_length=128)
            inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}

            with torch.no_grad():
                outputs1 = self.model(**inputs1)
                outputs2 = self.model(**inputs2)

            # 使用[CLS] token的特征
            feat1 = outputs1.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            feat2 = outputs2.last_hidden_state[:, 0, :].cpu().numpy().flatten()

            # 计算余弦相似度
            norm1 = np.linalg.norm(feat1)
            norm2 = np.linalg.norm(feat2)
            if norm1 == 0 or norm2 == 0:
                return 0.0

            raw_similarity = np.dot(feat1, feat2) / (norm1 * norm2)

            # 关键修复：将numpy类型转换为Python原生float
            raw_similarity = float(raw_similarity)

            print(f"📊 BERT原始相似度: {raw_similarity:.4f} (文本1: '{text1[:30]}...' 文本2: '{text2[:30]}...')")

            # ========== 修复：更严格的评分映射 ==========
            # BERT对中文短文本相似度通常偏高，需要保守映射

            if raw_similarity < 0.4:
                # 0.0-0.4: 完全不相关 -> 0-20分
                adjusted = raw_similarity / 0.4 * 0.2
            elif raw_similarity < 0.65:
                # 0.4-0.65: 可能相关 -> 20-50分
                adjusted = 0.2 + (raw_similarity - 0.4) / (0.65 - 0.4) * 0.3
            elif raw_similarity < 0.8:
                # 0.65-0.8: 相关 -> 50-75分
                adjusted = 0.5 + (raw_similarity - 0.65) / (0.8 - 0.65) * 0.25
            elif raw_similarity < 0.9:
                # 0.8-0.9: 高度相关 -> 75-90分
                adjusted = 0.75 + (raw_similarity - 0.8) / (0.9 - 0.8) * 0.15
            elif raw_similarity < 0.95:
                # 0.9-0.95: 非常相关 -> 90-95分
                adjusted = 0.9 + (raw_similarity - 0.9) / (0.95 - 0.9) * 0.05
            else:
                # >0.95: 几乎相同 -> 95-100分
                adjusted = 0.95 + (raw_similarity - 0.95) / (1.0 - 0.95) * 0.05

            # 确保分数在0-1之间
            adjusted = max(0.0, min(1.0, adjusted))
            print(f"🎯 BERT调整后相似度: {adjusted:.4f}")
            return float(adjusted)

        except Exception as e:
            print(f"BERT相似度计算错误: {e}")
            return float(self.simple_similarity(text1, text2))

    def simple_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        try:
            words1 = set(str(text1).lower().split())
            words2 = set(str(text2).lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return float(intersection / union if union > 0 else 0.0)
        except:
            return 0.0


# ==================== AI智能匹配引擎 ====================
class AIMatcher:
    def __init__(self):
        self.bert_matcher = BertMatcher()
        print("AI智能匹配引擎初始化完成")

    # 【新增】检查时间是否冲突
    def is_time_conflict(self, lost_date_str, found_date_str):
        if not lost_date_str or not found_date_str:
            return False
        try:
            # 尝试解析时间字符串
            t1 = parser.parse(str(lost_date_str))
            t2 = parser.parse(str(found_date_str))
            
            # 如果 拾获时间(t2) < 丢失时间(t1)，则是冲突
            # 比如：10号丢的，不可能9号就捡到了
            if t2 < t1:
                return True
            return False
        except Exception as e:
            print(f"时间解析错误: {e} (可能是格式不支持，跳过时间检查)")
            return False

    def match_items(self, lost_item: dict, found_item: dict) -> dict:
        try:
            # 【新增】1. 优先进行时间逻辑检查（硬约束）
            lost_date = lost_item.get('lost_date')
            found_date = found_item.get('found_date')
            
            if self.is_time_conflict(lost_date, found_date):
                print(f"⚠️ 时间逻辑冲突: 丢失({lost_date}) > 拾获({found_date})")
                return {
                    'match_score': 0.0,
                    'bert_score': 0.0,
                    'qianwen_analysis': f"时间逻辑冲突：拾获时间({found_date}) 早于 丢失时间({lost_date})，这在逻辑上是不可能的。",
                    'match_level': "不匹配"
                }

            # 调试信息
            print(f"\n🔍 匹配详情:")
            print(f"  丢失物品标题: '{lost_item.get('title')}'")
            print(f"  丢失物品描述: '{lost_item.get('description')}'")
            print(f"  丢失物品类别: '{lost_item.get('category')}'")
            print(f"  招领物品标题: '{found_item.get('title')}'")
            print(f"  招领物品描述: '{found_item.get('description')}'")
            print(f"  招领物品类别: '{found_item.get('category')}'")

            # ========== 方案2：分别计算不同部分的相似度 ==========
            print(f"\n📊 开始分项计算相似度:")

            # 1. 标题相似度（权重25%）
            title_sim = self.bert_matcher.calculate_similarity(
                lost_item.get('title', ''),
                found_item.get('title', '')
            )

            # 2. 描述相似度（权重50%）
            desc_sim = self.bert_matcher.calculate_similarity(
                lost_item.get('description', ''),
                found_item.get('description', '')
            )

            # 3. 类别相似度（权重25%）
            category_sim = self.bert_matcher.calculate_similarity(
                lost_item.get('category', ''),
                found_item.get('category', '')
            )

            # 加权综合
            bert_score = (title_sim * 0.1 + desc_sim * 0.8 + category_sim * 0.1) * 100

            print(f"\n🎯 分项相似度结果:")
            print(f"  标题相似度: {title_sim:.4f} (权重10%) -> {title_sim*100:.1f}分")
            print(f"  描述相似度: {desc_sim:.4f} (权重80%) -> {desc_sim*100:.1f}分")
            print(f"  类别相似度: {category_sim:.4f} (权重10%) -> {category_sim*100:.1f}分")
            print(f"  加权综合BERT分数: {bert_score:.1f}/100")

            # 通义千问分析
            qianwen_analysis = self.get_qianwen_analysis(lost_item, found_item, bert_score)
            print(f"  通义千问分析摘要: {qianwen_analysis[:50]}...")

            # 根据通义千问分析调整分数
            final_score = self.adjust_score_by_qianwen(bert_score, qianwen_analysis, lost_item, found_item)

            print(f"  最终匹配分数: {final_score:.1f}/100")

            return {
                'match_score': round(final_score, 1),
                'bert_score': round(bert_score, 1),
                'qianwen_analysis': qianwen_analysis,
                'match_level': self.get_match_level(final_score)
            }

        except Exception as e:
            print(f"匹配错误: {e}")
            return {
                'match_score': 0.0,
                'bert_score': 0.0,
                'qianwen_analysis': f"匹配失败: {str(e)}",
                'match_level': "错误"
            }

    def adjust_score_by_qianwen(self, bert_score: float, analysis: str, lost_item: dict, found_item: dict) -> float:
        """根据通义千问的分析调整分数"""
        original_score = bert_score
        adjusted_score = bert_score

        # 1. 检查分析中的否定词（强烈否定）
        strong_negative_keywords = [
            '不可能', '不可能是', '肯定不是', '绝对不是', '完全不同',
            '毫无关系', '没有关联', '不是同一个'
        ]

        for keyword in strong_negative_keywords:
            if keyword in analysis:
                print(f"  ⚠️ 检测到强烈否定词: '{keyword}'，大幅降低分数")
                reduction = 0.7 if bert_score > 60 else 0.5
                adjusted_score = bert_score * (1 - reduction)
                break

        # 2. 检查分析中的温和否定词
        if adjusted_score == bert_score:  # 如果还没被调整
            mild_negative_keywords = [
                '不太可能', '可能性小', '需要确认', '需要核实', '可能不同',
                '存在差异', '不一致', '有疑问'
            ]

            for keyword in mild_negative_keywords:
                if keyword in analysis:
                    print(f"  ⚠️ 检测到温和否定词: '{keyword}'，适当降低分数")
                    reduction = 0.3 if bert_score > 70 else 0.2
                    adjusted_score = bert_score * (1 - reduction)
                    break

        # 3. 检查分析中的肯定词
        if adjusted_score == bert_score:  # 如果还没被调整
            positive_keywords = [
                '可能匹配', '很可能', '高度相似', '非常相似', '建议联系',
                '可能是', '匹配度高', '相似度高'
            ]

            for keyword in positive_keywords:
                if keyword in analysis:
                    print(f"  ✅ 检测到肯定词: '{keyword}'，适当提高分数")
                    boost = 0.1 if bert_score < 80 else 0.05
                    adjusted_score = bert_score * (1 + boost)
                    # 限制最高分，防止超过100
                    adjusted_score = min(95.0, adjusted_score)  # 最高95分
                    break

        # 4. 特殊情况：如果BERT分数很高但物品明显不同，强制降低
        if bert_score > 70 and self.is_obviously_mismatch(lost_item, found_item):
            print(f"  ⚠️ BERT高分但物品明显不同，强制降低分数")
            adjusted_score = min(adjusted_score, 30.0)  # 最高30分

        # 5. 防止分数超过100或低于0
        adjusted_score = max(0.0, min(100.0, adjusted_score))

        # 6. 对于超过95分的情况，特别处理
        if adjusted_score > 95:
            print(f"  ⚠️ 分数过高({adjusted_score:.1f})，进行最终调整")
            # 检查是否有任何否定词，即使之前没匹配到
            if any(keyword in analysis for keyword in ['不同', '差异', '不一致']):
                adjusted_score = max(adjusted_score * 0.8, 85.0)  # 降低但保持较高分

        if abs(adjusted_score - original_score) > 1.0:
            print(f"  分数调整: {original_score:.1f} → {adjusted_score:.1f}")

        return float(adjusted_score)

    def is_obviously_mismatch(self, lost_item: dict, found_item: dict) -> bool:
        """检查是否明显不匹配"""
        lost_title = lost_item.get('title', '').lower()
        found_title = found_item.get('title', '').lower()

        # 定义明显不同的物品类型
        mismatched_pairs = [
            ('钥匙', '校园卡'), ('钥匙', '学生卡'), ('钥匙', '一卡通'),
            ('手机', '书包'), ('手机', '书本'), ('手机', '水杯'),
            ('钱包', '眼镜'), ('钱包', '充电器'), ('钱包', '衣服'),
            ('书本', '充电宝'), ('书本', '耳机'), ('书本', '雨伞')
        ]

        for lost_type, found_type in mismatched_pairs:
            if (lost_type in lost_title and found_type in found_title) or \
               (found_type in lost_title and lost_type in found_title):
                return True

        return False

    def get_qianwen_analysis(self, lost_item: dict, found_item: dict, bert_score: float) -> str:
        try:
            prompt = f"""
            请客观分析这两个物品是否可能匹配：

            当前BERT相似度评分：{bert_score:.1f}/100

            丢失物品信息：
            - 名称：{lost_item.get('title', '无')}
            - 描述：{lost_item.get('description', '无')}
            - 类别：{lost_item.get('category', '无')}
            - 颜色：{lost_item.get('color', '无')}
            - 品牌：{lost_item.get('brand', '无')}
            - 丢失地点：{lost_item.get('lost_location', '无')}
            - 丢失时间：{lost_item.get('lost_date', '无')}

            招领物品信息：
            - 名称：{found_item.get('title', '无')}
            - 描述：{found_item.get('description', '无')}
            - 类别：{found_item.get('category', '无')}
            - 颜色：{found_item.get('color', '无')}
            - 品牌：{found_item.get('brand', '无')}
            - 拾获地点：{found_item.get('found_location', '无')}
            - 拾获时间：{found_item.get('found_date', '无')}

            请从以下几个方面分析：
            1. 物品名称、类别、描述是否一致或相似
            2. 关键特征（颜色、品牌等）是否匹配
            3. 丢失和拾获的地点、时间是否有相关性
            4. 给出最终判断：是否可能是同一个物品

            请用简洁客观的语言分析，不要重复BERT分数。
            """

            response = qianwen.chat(prompt, [])
            return response.strip()[:300]

        except Exception as e:
            print(f"通义千问分析失败: {e}")
            return "AI分析暂不可用"

    def get_match_level(self, score: float) -> str:
        # 调整匹配等级阈值
        if score >= 75:
            return "高度匹配"
        elif score >= 55:
            return "中度匹配"
        elif score >= 35:
            return "轻度匹配"
        elif score >= 15:
            return "可能相关"
        else:
            return "不匹配"


# 初始化AI匹配引擎
print("启动校园失物招领AI匹配平台")
ai_matcher = AIMatcher()


# ==================== 路由定义 ====================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/lost')
def show_lost():
    return render_template('lost_items.html', items=lost_items)


@app.route('/found')
def show_found():
    return render_template('found_items.html', items=found_items)


@app.route('/match')
def show_match():
    return render_template('match.html')


@app.route('/ai/assistant')
def ai_assistant():
    return render_template('ai_assistant.html')


@app.route('/submit/lost')
def submit_lost_page():
    return render_template('submit_lost.html')


@app.route('/submit/found')
def submit_found_page():
    return render_template('submit_found.html')


@app.route('/api/submit/lost', methods=['POST'])
def submit_lost():
    try:
        data = request.json
        item_data = {
            'id': str(uuid.uuid4())[:8],
            'title': data.get('title', ''),
            'description': data.get('description', ''),
            'category': data.get('category', ''),
            'color': data.get('color', ''),
            'brand': data.get('brand', ''),
            'lost_date': data.get('lost_date', ''),
            'lost_location': data.get('lost_location', ''),
            'contact_name': data.get('contact_name', ''),
            'contact_phone': data.get('contact_phone', ''),
            'contact_email': data.get('contact_email', ''),
            'image_url': data.get('image_url', ''),
            'status': '寻找中',
            'created_at': datetime.datetime.now().isoformat()
        }
        lost_items.append(item_data)
        return jsonify({'success': True, 'item_id': item_data['id'], 'message': '丢失物品已提交'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/submit/found', methods=['POST'])
def submit_found():
    try:
        data = request.json
        item_data = {
            'id': str(uuid.uuid4())[:8],
            'title': data.get('title', ''),
            'description': data.get('description', ''),
            'category': data.get('category', ''),
            'color': data.get('color', ''),
            'brand': data.get('brand', ''),
            'found_date': data.get('found_date', ''),
            'found_location': data.get('found_location', ''),
            'contact_name': data.get('contact_name', ''),
            'contact_phone': data.get('contact_phone', ''),
            'contact_email': data.get('contact_email', ''),
            'image_url': data.get('image_url', ''),
            'status': '待认领',
            'created_at': datetime.datetime.now().isoformat()
        }
        found_items.append(item_data)
        return jsonify({'success': True, 'item_id': item_data['id'], 'message': '招领物品已提交'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/items/lost', methods=['GET'])
def get_lost_items():
    return jsonify({'items': lost_items, 'count': len(lost_items)})


@app.route('/api/items/found', methods=['GET'])
def get_found_items():
    return jsonify({'items': found_items, 'count': len(found_items)})


@app.route('/api/item/<item_id>', methods=['GET'])
def get_item(item_id):
    for item in lost_items:
        if item['id'] == item_id:
            return jsonify({'item': item, 'type': 'lost'})
    for item in found_items:
        if item['id'] == item_id:
            return jsonify({'item': item, 'type': 'found'})
    return jsonify({'error': '物品不存在'}), 404

@app.route('/lost/<item_id>')
def lost_item_detail(item_id):
    for item in lost_items:
        if item['id'] == item_id:
            return render_template('lost_item_detail.html', item=item)
    return "物品不存在", 404

@app.route('/found/<item_id>')
def found_item_detail(item_id):
    for item in found_items:
        if item['id'] == item_id:
            return render_template('found_item_detail.html', item=item)
    return "物品不存在", 404

@app.route('/api/match', methods=['POST'])
def match_items():
    """
    兼容性匹配API - 转发到深度匹配
    支持两种参数格式:
    1. {lost_item_id, found_item_id} - 精确匹配两个物品
    2. {found_item_id} - 搜索匹配的丢失物品
    3. {lost_item_id} - 搜索匹配的招领物品
    """
    try:
        data = request.json
        lost_item_id = data.get('lost_item_id')
        found_item_id = data.get('found_item_id')

        print(f"\n📞 收到匹配请求 (兼容模式):")
        print(f"   参数: {data}")

        # 模式1: 精确匹配两个物品
        if lost_item_id and found_item_id:
            print("   模式: 精确匹配两个物品")

            # 查找物品
            lost_item = None
            found_item = None

            for item in lost_items:
                if item['id'] == lost_item_id:
                    lost_item = item
                    break

            for item in found_items:
                if item['id'] == found_item_id:
                    found_item = item
                    break

            if not lost_item or not found_item:
                return jsonify({'error': '物品不存在'}), 404

            # 进行匹配
            match_result = ai_matcher.match_items(lost_item, found_item)

            # 保存匹配记录
            match_record = {
                'id': str(uuid.uuid4())[:8],
                'lost_item_id': lost_item_id,
                'found_item_id': found_item_id,
                'match_result': match_result,
                'timestamp': datetime.datetime.now().isoformat()
            }
            matches.append(match_record)

            return jsonify({
                'success': True,
                'match_id': match_record['id'],
                'match_result': match_result,
                'lost_item': {
                    'id': lost_item['id'],
                    'title': lost_item['title'],
                    'description': lost_item['description']
                },
                'found_item': {
                    'id': found_item['id'],
                    'title': found_item['title'],
                    'description': found_item['description']
                }
            })

        # 模式2: 只提供招领物品，搜索所有丢失物品
        elif found_item_id and not lost_item_id:
            print("   模式: 搜索匹配的丢失物品")

            # 查找招领物品
            found_item = None
            for item in found_items:
                if item['id'] == found_item_id:
                    found_item = item
                    break

            if not found_item:
                return jsonify({'error': '招领物品不存在'}), 404

            # 与所有丢失物品进行匹配
            all_matches = []
            for lost_item in lost_items:
                try:
                    match_result = ai_matcher.match_items(lost_item, found_item)

                    # 只保留分数较高的匹配
                    if match_result['match_score'] > 20:
                        all_matches.append({
                            'lost_item': {
                                'id': lost_item['id'],
                                'title': lost_item['title'],
                                'description': lost_item['description']
                            },
                            'match_result': match_result
                        })

                except Exception as e:
                    print(f"匹配失败: {lost_item['id']} - {e}")

            # 按分数排序
            all_matches.sort(key=lambda x: x['match_result']['match_score'], reverse=True)

            return jsonify({
                'success': True,
                'found_item': {
                    'id': found_item['id'],
                    'title': found_item['title'],
                    'description': found_item['description']
                },
                'matches': all_matches[:10],
                'total_matches': len(all_matches)
            })

        # 模式3: 只提供丢失物品，搜索所有招领物品
        elif lost_item_id and not found_item_id:
            print("   模式: 搜索匹配的招领物品")

            # 查找丢失物品
            lost_item = None
            for item in lost_items:
                if item['id'] == lost_item_id:
                    lost_item = item
                    break

            if not lost_item:
                return jsonify({'error': '丢失物品不存在'}), 404

            # 与所有招领物品进行匹配
            all_matches = []
            for found_item in found_items:
                try:
                    match_result = ai_matcher.match_items(lost_item, found_item)

                    # 只保留分数较高的匹配
                    if match_result['match_score'] > 20:
                        all_matches.append({
                            'found_item': {
                                'id': found_item['id'],
                                'title': found_item['title'],
                                'description': found_item['description']
                            },
                            'match_result': match_result
                        })

                except Exception as e:
                    print(f"匹配失败: {found_item['id']} - {e}")

            # 按分数排序
            all_matches.sort(key=lambda x: x['match_result']['match_score'], reverse=True)

            return jsonify({
                'success': True,
                'lost_item': {
                    'id': lost_item['id'],
                    'title': lost_item['title'],
                    'description': lost_item['description']
                },
                'matches': all_matches[:10],
                'total_matches': len(all_matches)
            })

        else:
            return jsonify({'error': '需要提供至少一个物品ID'}), 400

    except Exception as e:
        print(f"❌ 兼容匹配API错误: {e}")
        return jsonify({'error': f'匹配失败: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'success': True, 'filename': filename, 'url': f'/uploads/{filename}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = {
        'total_lost': len(lost_items),
        'total_found': len(found_items),
        'total_matches': len(matches),
        'bert_model_loaded': ai_matcher.bert_matcher.model_loaded,
        'server_time': datetime.datetime.now().isoformat()
    }
    return jsonify(stats)


@app.route('/api/ai/describe', methods=['POST'])
def ai_describe_item():
    try:
        data = request.json
        item_type = data.get('item_type', '').strip()
        features = data.get('features', '').strip()
        if not item_type:
            return jsonify({'error': '请输入物品类型'}), 400
        description = qianwen.generate_item_description(item_type, features)
        return jsonify({'description': description})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        history = data.get('history', [])
        if not user_message:
            return jsonify({'error': '请输入消息'}), 400
        ai_reply = qianwen.chat(user_message, history)
        return jsonify({'reply': ai_reply, 'timestamp': datetime.datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/notice', methods=['POST'])
def ai_generate_notice():
    try:
        data = request.json
        item_info = data.get('item_info', {})
        notice_type = data.get('notice_type', 'lost')
        if not item_info:
            return jsonify({'error': '请提供物品信息'}), 400
        notice = qianwen.generate_notice(item_info, notice_type)
        return jsonify({'notice': notice, 'notice_type': notice_type})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '路由不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500


if __name__ == '__main__':
    print(f"服务已启动: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)