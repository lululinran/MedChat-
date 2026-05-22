"""智能检索 Agent - 自动判断检索策略并支持迭代优化"""
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
import re

# 检索策略枚举
class RetrievalStrategy(Enum):
    VECTOR_ONLY = "vector_only"       # 仅向量检索
    GRAPH_ONLY = "graph_only"         # 仅图谱检索
    HYBRID = "hybrid"                 # 混合检索
    AUTO = "auto"                     # 自动选择

# 问题类型枚举
class QuestionType(Enum):
    FACTUAL = "factual"               # 事实性问题（如：糖尿病的症状是什么）
    EXPLANATORY = "explanatory"       # 解释性问题（如：为什么会得糖尿病）
    ADVISORY = "advisory"             # 建议性问题（如：我该怎么办）
    COMPARATIVE = "comparative"       # 比较性问题（如：A和B有什么区别）
    PROCEDURAL = "procedural"         # 过程性问题（如：如何治疗）
    UNKNOWN = "unknown"               # 未知类型

class SmartRetrievalAgent:
    """智能检索 Agent - 自动判断检索策略并支持迭代反思"""
    
    def __init__(self):
        # 导入医疗 NER
        from .medical_ner import get_medical_ner
        self.ner = get_medical_ner()
        
        # 有效实体类型列表
        self.valid_entity_types = {'DISEASE', 'SYMPTOM', 'DRUG', 'TREATMENT', 'CHECK'}
        
        # 意图关键词模式（增强版）
        self.intent_patterns = {
            QuestionType.FACTUAL: [
                r'是什么', r'有哪些', r'什么是', r'包括', r'包含',
                r'定义', r'概念', r'含义', r'意思', r'指的是',
                r'症状', r'病因', r'治疗', r'诊断', r'检查',
                r'预防', r'并发症', r'预后', r'发病率', r'死亡率',
                r'常用药品', r'推荐药品', r'宜吃', r'忌吃', r'所属科室'
            ],
            QuestionType.EXPLANATORY: [
                r'为什么', r'原因', r'为什么会', r'为何', r'机理',
                r'原理', r'机制', r'如何形成', r'怎么引起', r'诱因',
                r'由来', r'根源', r'背景', r'因素', r'导致'
            ],
            QuestionType.ADVISORY: [
                r'怎么办', r'该怎么做', r'如何处理', r'建议', r'推荐',
                r'应该', r'需要', r'可以', r'最好', r'适合',
                r'帮忙', r'指导', r'请教', r'支招', r'如何应对'
            ],
            QuestionType.COMPARATIVE: [
                r'区别', r'对比', r'比较', r'差异', r'不同',
                r'vs', r'和', r'与', r'哪个好', r'哪个更',
                r'相比', r'对照', r'优劣', r'异同', r'差别'
            ],
            QuestionType.PROCEDURAL: [
                r'如何', r'步骤', r'流程', r'方法', r'过程',
                r'怎么做', r'操作', r'实施', r'执行', r'进行',
                r'步骤', r'流程', r'指南', r'教程', r'步骤'
            ]
        }
        
        # 置信度阈值
        self.confidence_threshold = 0.75
        self.max_iterations = 3
    
    def extract_entities(self, question: str) -> List[Dict]:
        """
        提取医疗实体，优先使用规则匹配（更可靠）
        """
        # 先尝试使用模型
        entities = self.ner.extract_entities(question)
        
        # 检查模型识别结果是否有效
        valid_entities = [e for e in entities if e['entity_type'] in self.valid_entity_types]
        
        # 如果模型识别结果不好，使用规则匹配
        if len(valid_entities) == 0 or all(e['score'] < 0.7 for e in entities):
            # 强制使用规则匹配
            original_ner = self.ner.ner
            self.ner.ner = None
            entities = self.ner.extract_entities(question)
            self.ner.ner = original_ner
        
        return entities
    
    def classify_question(self, question: str) -> Tuple[QuestionType, float]:
        """
        分类用户问题类型
        
        Returns:
            (问题类型, 置信度)
        """
        scores = {q_type: 0 for q_type in QuestionType}
        
        # 基于关键词的规则匹配（增强版）
        # 每个匹配增加分数
        for q_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    scores[q_type] += 1
        
        # 特殊模式检测
        # "什么"开头更可能是事实性问题
        if question.startswith('什么'):
            scores[QuestionType.FACTUAL] += 2
        
        # "为什么"开头是解释性问题
        if question.startswith('为什么'):
            scores[QuestionType.EXPLANATORY] += 3
        
        # "怎么"开头可能是建议性或过程性问题
        if question.startswith('怎么'):
            if '怎么办' in question or '处理' in question:
                scores[QuestionType.ADVISORY] += 3
            else:
                scores[QuestionType.PROCEDURAL] += 2
        
        # "如何"开头通常是过程性问题
        if question.startswith('如何'):
            scores[QuestionType.PROCEDURAL] += 3
        
        # "有什么区别"是比较性问题
        if '区别' in question or '差异' in question:
            scores[QuestionType.COMPARATIVE] += 3
        
        # "什么区别"是比较性问题
        if '什么区别' in question or '什么不同' in question:
            scores[QuestionType.COMPARATIVE] += 3
        
        # "该"、"应该"、"可以"倾向于建议性问题
        advisory_words = ['该', '应该', '可以', '建议', '推荐', '最好']
        for word in advisory_words:
            if word in question:
                scores[QuestionType.ADVISORY] += 1
        
        # 归一化分数
        max_possible = max(len(patterns) for patterns in self.intent_patterns.values()) + 5
        normalized_scores = {k: v / max_possible for k, v in scores.items()}
        
        max_type = max(normalized_scores, key=normalized_scores.get)
        max_score = normalized_scores[max_type]
        
        # 阈值判断
        if max_score < 0.15:
            return QuestionType.UNKNOWN, max_score
        return max_type, max_score
    
    def determine_strategy(self, question: str, entities: List[Dict]) -> RetrievalStrategy:
        """
        根据问题类型和实体信息决定检索策略
        
        Strategy Logic:
        - 事实性问题 + 明确医疗实体 → 优先图谱检索
        - 解释性/建议性问题 → 混合检索
        - 比较性问题 → 混合检索
        - 无明确实体 → 优先向量检索
        """
        q_type, confidence = self.classify_question(question)
        
        # 提取实体类型统计
        entity_types = [e['entity_type'] for e in entities]
        has_disease = 'DISEASE' in entity_types
        has_symptom = 'SYMPTOM' in entity_types
        has_drug = 'DRUG' in entity_types
        
        # 策略决策逻辑
        if q_type == QuestionType.FACTUAL:
            if has_disease and confidence > 0.5:
                # 事实性问题 + 明确疾病 → 优先图谱
                return RetrievalStrategy.GRAPH_ONLY
            elif len(entities) > 0:
                # 有实体但置信度不高 → 混合检索
                return RetrievalStrategy.HYBRID
            else:
                # 无实体 → 向量检索
                return RetrievalStrategy.VECTOR_ONLY
        
        elif q_type in [QuestionType.EXPLANATORY, QuestionType.ADVISORY]:
            # 解释性/建议性问题 → 混合检索
            return RetrievalStrategy.HYBRID
        
        elif q_type == QuestionType.COMPARATIVE:
            # 比较性问题 → 混合检索
            return RetrievalStrategy.HYBRID
        
        elif q_type == QuestionType.PROCEDURAL:
            if has_disease or has_drug:
                return RetrievalStrategy.HYBRID
            else:
                return RetrievalStrategy.VECTOR_ONLY
        
        else:
            # 未知类型 → 混合检索
            return RetrievalStrategy.HYBRID
    
    def analyze_answer_quality(self, answer: str, question: str, entities: List[Dict]) -> float:
        """
        分析答案质量，返回置信度分数
        
        评估维度：
        1. 是否包含问题中的实体
        2. 是否直接回答了问题
        3. 是否提供了具体信息
        4. 是否有矛盾或不确定表述
        """
        score = 0.0
        factors = []
        
        # 检查是否包含关键实体
        entity_names = [e['entity'] for e in entities]
        entity_matches = sum(1 for entity in entity_names if entity in answer)
        if entity_matches > 0:
            score += 0.3
            factors.append(f"包含 {entity_matches} 个实体")
        else:
            score -= 0.2
            factors.append("未提及实体")
        
        # 检查是否直接回答问题
        question_keywords = ['什么', '怎么', '为什么', '如何', '哪里', '多少']
        answer_contains_action = any(k in answer for k in ['建议', '应该', '可以', '需要', '不要'])
        if answer_contains_action or any(k in question for k in ['怎么办', '如何']):
            score += 0.2
            factors.append("提供了行动建议")
        
        # 检查答案长度和信息量
        if len(answer) > 100:
            score += 0.2
            factors.append("答案详细")
        elif len(answer) < 20:
            score -= 0.1
            factors.append("答案简短")
        
        # 检查是否有不确定表述
        uncertain_phrases = ['可能', '也许', '大概', '不确定', '不太清楚', '无法确定']
        if any(phrase in answer for phrase in uncertain_phrases):
            score -= 0.1
            factors.append("存在不确定表述")
        
        # 检查是否有拒绝回答
        refusal_phrases = ['无法回答', '不知道', '不清楚', '没有相关信息']
        if any(phrase in answer for phrase in refusal_phrases):
            score -= 0.3
            factors.append("存在拒绝回答")
        
        # 归一化到 0-1
        final_score = max(0.0, min(1.0, score + 0.4))
        
        return final_score, factors
    
    def reflect_and_iterate(self, question: str, answer: str, entities: List[Dict], 
                           strategy: RetrievalStrategy, iteration: int = 1) -> Dict:
        """
        反思答案质量并决定是否需要迭代检索
        
        Returns:
            {
                'need_iteration': bool,
                'new_strategy': RetrievalStrategy,
                'reason': str,
                'quality_score': float
            }
        """
        quality_score, factors = self.analyze_answer_quality(answer, question, entities)
        
        if iteration >= self.max_iterations:
            return {
                'need_iteration': False,
                'new_strategy': strategy,
                'reason': f"已达到最大迭代次数 ({self.max_iterations})",
                'quality_score': quality_score,
                'factors': factors
            }
        
        if quality_score >= self.confidence_threshold:
            return {
                'need_iteration': False,
                'new_strategy': strategy,
                'reason': f"答案质量达标 ({quality_score:.2f})",
                'quality_score': quality_score,
                'factors': factors
            }
        
        # 需要迭代，决定新策略
        new_strategy = strategy
        reason = ""
        
        if strategy == RetrievalStrategy.GRAPH_ONLY:
            # 图谱检索不够，尝试混合检索
            new_strategy = RetrievalStrategy.HYBRID
            reason = "图谱检索结果不足，尝试混合检索"
        elif strategy == RetrievalStrategy.VECTOR_ONLY:
            # 向量检索不够，尝试混合检索
            new_strategy = RetrievalStrategy.HYBRID
            reason = "向量检索结果不足，尝试混合检索"
        elif strategy == RetrievalStrategy.HYBRID:
            # 混合检索不够，尝试扩大检索范围或调整参数
            new_strategy = RetrievalStrategy.HYBRID
            reason = "混合检索结果仍不足，尝试扩大检索范围"
        
        return {
            'need_iteration': True,
            'new_strategy': new_strategy,
            'reason': reason,
            'quality_score': quality_score,
            'factors': factors
        }
    
    def retrieve(self, question: str, strategy: Optional[RetrievalStrategy] = None) -> Dict:
        """
        执行智能检索
        
        Returns:
            {
                'strategy': RetrievalStrategy,
                'entities': List[Dict],
                'question_type': QuestionType,
                'retrieval_results': Dict,
                'iterations': int
            }
        """
        # 1. 提取实体
        entities = self.extract_entities(question)
        
        # 2. 如果未指定策略，自动决定
        if strategy is None:
            strategy = self.determine_strategy(question, entities)
        
        # 3. 分类问题类型
        q_type, q_confidence = self.classify_question(question)
        
        # 4. 执行检索（模拟）
        results = {
            'vector_results': [],
            'graph_results': [],
            'strategy': strategy.value
        }
        
        if strategy in [RetrievalStrategy.VECTOR_ONLY, RetrievalStrategy.HYBRID]:
            results['vector_results'] = self._mock_vector_search(question, entities)
        
        if strategy in [RetrievalStrategy.GRAPH_ONLY, RetrievalStrategy.HYBRID]:
            results['graph_results'] = self._mock_graph_search(question, entities)
        
        return {
            'strategy': strategy,
            'entities': entities,
            'question_type': q_type,
            'question_confidence': q_confidence,
            'retrieval_results': results
        }
    
    def _mock_vector_search(self, question: str, entities: List[Dict]) -> List[Dict]:
        """模拟向量检索结果"""
        mock_results = [
            {'content': f"关于{entities[0]['entity']}的详细信息...", 'score': 0.92},
            {'content': f"{entities[0]['entity']}的最新研究进展...", 'score': 0.87},
            {'content': f"临床指南：{entities[0]['entity']}的诊疗规范...", 'score': 0.81}
        ] if entities else [
            {'content': "相关医疗知识文档...", 'score': 0.75}
        ]
        return mock_results
    
    def _mock_graph_search(self, question: str, entities: List[Dict]) -> List[Dict]:
        """模拟图谱检索结果"""
        if not entities:
            return []
        
        mock_results = []
        for entity in entities:
            if entity['entity_type'] == 'DISEASE':
                mock_results.append({
                    'entity': entity['entity'],
                    'relations': [
                        {'relation': '症状', 'value': '发热、头痛、乏力'},
                        {'relation': '治疗', 'value': '药物治疗、手术'},
                        {'relation': '预防', 'value': '接种疫苗、注意卫生'}
                    ]
                })
            elif entity['entity_type'] == 'SYMPTOM':
                mock_results.append({
                    'entity': entity['entity'],
                    'relations': [
                        {'relation': '相关疾病', 'value': '感冒、流感、肺炎'},
                        {'relation': '常见原因', 'value': '病毒感染、细菌感染'}
                    ]
                })
        return mock_results

# 全局实例
_smart_agent = None

def get_smart_retrieval_agent() -> SmartRetrievalAgent:
    """获取智能检索 Agent 实例"""
    global _smart_agent
    if _smart_agent is None:
        _smart_agent = SmartRetrievalAgent()
    return _smart_agent

if __name__ == "__main__":
    # 测试
    agent = get_smart_retrieval_agent()
    
    test_cases = [
        "糖尿病的症状是什么",
        "我牙龈肿了怎么办",
        "高血压和糖尿病有什么区别",
        "为什么会得心脏病",
        "如何治疗感冒"
    ]
    
    for question in test_cases:
        print(f"\n{'='*50}")
        print(f"问题: {question}")
        
        # 分类问题
        q_type, q_conf = agent.classify_question(question)
        print(f"问题类型: {q_type.value} (置信度: {q_conf:.2f})")
        
        # 提取实体
        entities = agent.extract_entities(question)
        print(f"提取实体: {entities}")
        
        # 决定策略
        strategy = agent.determine_strategy(question, entities)
        print(f"检索策略: {strategy.value}")
        
        # 执行检索
        result = agent.retrieve(question, strategy)
        print(f"检索结果: {json.dumps(result['retrieval_results'], ensure_ascii=False, indent=2)}")
        
        # 模拟答案和反思
        mock_answer = "根据检索结果，糖尿病的常见症状包括多饮、多食、多尿、体重减轻等。"
        quality, factors = agent.analyze_answer_quality(mock_answer, question, entities)
        print(f"答案质量: {quality:.2f}")
        print(f"评估因素: {factors}")