"""智能检索工具 - 集成智能Agent决策和双向量数据库"""
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json

# 导入依赖
from .smart_retrieval_agent import (
    get_smart_retrieval_agent, 
    RetrievalStrategy, 
    QuestionType
)
from .dual_vector_store import (
    search_episodic_memory, 
    search_semantic_memory, 
    search_hybrid_memory,
    VectorStoreType
)
from .tools import search_medical_graph, emit_rag_step
from .medical_ner import get_medical_ner

class RetrievalMode(Enum):
    """检索模式"""
    AUTO = "auto"                 # 自动选择
    EPISODIC_ONLY = "episodic"    # 仅情景记忆
    SEMANTIC_ONLY = "semantic"    # 仅语义记忆
    GRAPH_ONLY = "graph"          # 仅知识图谱
    VECTOR_ONLY = "vector_only"   # 仅向量检索（情景+语义，不含图谱）
    HYBRID_FULL = "hybrid_full"   # 全混合（情景+语义+图谱）

class IntelligentRetrieval:
    """智能检索器 - 基于Agent决策的多源检索"""
    
    def __init__(self):
        self.agent = get_smart_retrieval_agent()
        self.ner = get_medical_ner()
    
    def _extract_entities_with_rules(self, query: str) -> List[Dict]:
        """使用规则匹配提取实体（更可靠）"""
        # 强制使用规则匹配
        original_ner = self.ner.ner
        self.ner.ner = None
        entities = self.ner.extract_entities(query)
        self.ner.ner = original_ner
        return entities
    
    def analyze_query(self, query: str) -> Dict:
        """
        分析用户查询，返回分析结果
        
        Returns:
            {
                'question_type': QuestionType,
                'entities': List[Dict],
                'strategy': RetrievalStrategy,
                'confidence': float
            }
        """
        # 分类问题类型
        q_type, q_conf = self.agent.classify_question(query)
        
        # 提取实体（使用规则匹配）
        entities = self._extract_entities_with_rules(query)
        
        # 决定检索策略
        strategy = self.agent.determine_strategy(query, entities)
        
        return {
            'question_type': q_type,
            'question_confidence': q_conf,
            'entities': entities,
            'strategy': strategy,
            'query': query
        }
    
    def retrieve(self, query: str, mode: Optional[RetrievalMode] = None) -> Dict:
        """
        执行智能检索
        
        Args:
            query: 用户查询
            mode: 检索模式（默认自动选择）
        
        Returns:
            {
                'strategy': str,
                'question_type': str,
                'entities': List[Dict],
                'results': {
                    'episodic': List[Dict],
                    'semantic': List[Dict],
                    'graph': List[Dict]
                },
                'iterations': int,
                'quality_score': float
            }
        """
        # 分析查询
        analysis = self.analyze_query(query)
        entities = analysis['entities']
        q_type = analysis['question_type']
        
        emit_rag_step('🧠', '查询分析完成', f'问题类型: {q_type.value}, 实体: {[e["entity"] for e in entities]}')
        
        # 确定检索策略
        if mode is None:
            strategy = analysis['strategy']
            mode = self._strategy_to_mode(strategy)
        else:
            strategy = self._mode_to_strategy(mode)
        
        emit_rag_step('🎯', '确定检索策略', f'策略: {strategy.value}, 模式: {mode.value}')
        
        # 执行检索
        results = {
            'episodic': [],
            'semantic': [],
            'graph': []
        }
        
        # 根据模式检索不同数据源
        # EPISODIC_ONLY/VECTOR_ONLY/HYBRID_FULL 检索情景记忆（用户自己的病例）
        if mode in [RetrievalMode.EPISODIC_ONLY, RetrievalMode.VECTOR_ONLY, RetrievalMode.HYBRID_FULL]:
            emit_rag_step('📝', '检索用户病例', '查找您的就诊记录和检查报告...')
            episodic_results = search_episodic_memory(query)
            results['episodic'] = episodic_results
            emit_rag_step('✅', '用户病例检索完成', f'找到 {len(episodic_results)} 条相关记录')
        
        # SEMANTIC_ONLY/VECTOR_ONLY/HYBRID_FULL 检索语义记忆
        if mode in [RetrievalMode.SEMANTIC_ONLY, RetrievalMode.VECTOR_ONLY, RetrievalMode.HYBRID_FULL]:
            emit_rag_step('📚', '检索语义记忆', '查找药品说明书和医学文献...')
            semantic_results = search_semantic_memory(query)
            results['semantic'] = semantic_results
            emit_rag_step('✅', '语义记忆检索完成', f'找到 {len(semantic_results)} 条记录')
        
        # GRAPH_ONLY/HYBRID_FULL 检索知识图谱
        if mode in [RetrievalMode.GRAPH_ONLY, RetrievalMode.HYBRID_FULL]:
            emit_rag_step('🧬', '检索知识图谱', '查询结构化医疗知识...')
            graph_results = self._retrieve_graph(query, entities)
            results['graph'] = graph_results
            emit_rag_step('✅', '知识图谱检索完成', f'找到 {len(graph_results)} 条记录')
        
        # 评估答案质量（模拟）
        quality_score = self._evaluate_results(results)
        
        return {
            'strategy': strategy.value,
            'mode': mode.value,
            'question_type': q_type.value,
            'entities': entities,
            'results': results,
            'quality_score': quality_score
        }
    
    def _retrieve_graph(self, query: str, entities: List[Dict]) -> List[Dict]:
        """执行知识图谱检索"""
        graph_results = []
        
        if not entities:
            return graph_results
        
        # 获取意图列表
        intents = self._extract_intents(query)
        
        for entity in entities[:3]:  # 最多处理3个实体
            entity_name = entity['entity']
            for intent in intents[:2]:  # 最多处理2个意图
                try:
                    result = search_medical_graph.invoke({
                        'entity': entity_name,
                        'intent': intent
                    })
                    if result and "未返回可用结果" not in result:
                        graph_results.append({
                            'entity': entity_name,
                            'intent': intent,
                            'content': result,
                            'entity_type': entity['entity_type']
                        })
                except Exception as e:
                    print(f"⚠️ 图谱检索失败: {e}")
        
        return graph_results
    
    def _extract_intents(self, query: str) -> List[str]:
        """从查询中提取意图"""
        intent_keywords = [
            (r"症状|表现", "疾病症状"),
            (r"病因|原因", "疾病病因"),
            (r"治疗|怎么办", "治疗方法"),
            (r"药品|用药", "推荐药品"),
            (r"检查", "所需检查"),
            (r"预防", "预防措施"),
            (r"忌吃|忌口", "忌吃食物"),
            (r"宜吃|推荐吃", "宜吃食物"),
            (r"科室", "所属科室"),
        ]
        
        intents = []
        for pattern, intent in intent_keywords:
            if __import__('re').search(pattern, query):
                intents.append(intent)
        
        if not intents:
            intents.append("疾病简介")
        
        return intents
    
    def _strategy_to_mode(self, strategy: RetrievalStrategy) -> RetrievalMode:
        """将Agent策略转换为检索模式"""
        mapping = {
            RetrievalStrategy.VECTOR_ONLY: RetrievalMode.VECTOR_ONLY,  # 仅向量检索
            RetrievalStrategy.GRAPH_ONLY: RetrievalMode.GRAPH_ONLY,
            RetrievalStrategy.HYBRID: RetrievalMode.HYBRID_FULL,
            RetrievalStrategy.AUTO: RetrievalMode.AUTO
        }
        return mapping.get(strategy, RetrievalMode.HYBRID_FULL)
    
    def _mode_to_strategy(self, mode: RetrievalMode) -> RetrievalStrategy:
        """将检索模式转换为Agent策略"""
        mapping = {
            RetrievalMode.EPISODIC_ONLY: RetrievalStrategy.VECTOR_ONLY,
            RetrievalMode.SEMANTIC_ONLY: RetrievalStrategy.VECTOR_ONLY,
            RetrievalMode.GRAPH_ONLY: RetrievalStrategy.GRAPH_ONLY,
            RetrievalMode.HYBRID_FULL: RetrievalStrategy.HYBRID,
            RetrievalMode.AUTO: RetrievalStrategy.AUTO
        }
        return mapping.get(mode, RetrievalStrategy.HYBRID)
    
    def _evaluate_results(self, results: Dict) -> float:
        """评估检索结果质量"""
        total = 0
        count = 0
        
        for source, items in results.items():
            if items:
                total += len(items) * 0.3
                count += 1
        
        if count == 3:  # 三个来源都有结果
            return min(1.0, total / 3 + 0.3)
        elif count == 2:
            return min(1.0, total / 2 + 0.2)
        elif count == 1:
            return min(1.0, total + 0.1)
        return 0.0
    
    def format_results(self, retrieval_result: Dict) -> str:
        """格式化检索结果为可读文本"""
        sections = []
        results = retrieval_result['results']
        
        # 情景记忆
        if results['episodic']:
            episodic_text = "\n".join([
                f"- {r['source']}: {r['content'][:100]}..." 
                for r in results['episodic'][:3]
            ])
            sections.append(f"## 📝 患者病例记录\n{episodic_text}")
        
        # 语义记忆
        if results['semantic']:
            semantic_text = "\n".join([
                f"- {r['source']}: {r['content'][:100]}..." 
                for r in results['semantic'][:3]
            ])
            sections.append(f"## 📚 药品说明书\n{semantic_text}")
        
        # 知识图谱
        if results['graph']:
            graph_text = "\n".join([
                f"- **{r['entity']}** ({r['intent']}): {r['content'][:100]}..." 
                for r in results['graph'][:3]
            ])
            sections.append(f"## 🧬 医疗知识图谱\n{graph_text}")
        
        if not sections:
            return "未找到相关信息。"
        
        return "\n\n---\n\n".join(sections)

# 全局实例
_intelligent_retrieval = None

def get_intelligent_retrieval() -> IntelligentRetrieval:
    """获取智能检索器实例"""
    global _intelligent_retrieval
    if _intelligent_retrieval is None:
        _intelligent_retrieval = IntelligentRetrieval()
    return _intelligent_retrieval

# 便捷函数
def intelligent_search(query: str, mode: Optional[RetrievalMode] = None) -> Dict:
    """执行智能检索"""
    return get_intelligent_retrieval().retrieve(query, mode)

if __name__ == "__main__":
    # 测试
    retriever = get_intelligent_retrieval()
    
    test_queries = [
        "糖尿病的症状是什么",
        "我牙龈肿了怎么办",
        "阿莫西林的用法用量",
        "高血压患者的用药注意事项"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        
        # 分析
        analysis = retriever.analyze_query(query)
        print(f"问题类型: {analysis['question_type'].value}")
        print(f"实体: {analysis['entities']}")
        print(f"策略: {analysis['strategy'].value}")
        
        # 检索
        result = retriever.retrieve(query)
        print(f"\n检索结果:")
        print(retriever.format_results(result))