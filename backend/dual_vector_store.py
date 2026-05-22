"""双向量数据库架构 - 情景记忆 + 语义记忆"""
from typing import List, Dict, Optional
import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# 导入项目中的 embedding service
try:
    from .embedding import embedding_service
except ImportError:
    embedding_service = None

class VectorStoreType(Enum):
    """向量数据库类型枚举"""
    EPISODIC = "episodic"     # 情景记忆 - 用户上传的病例、就诊记录
    SEMANTIC = "semantic"      # 语义记忆 - 药品说明书、医学文献

class DualVectorStore:
    """双向量数据库管理器"""
    
    def __init__(self):
        # 情景记忆数据库配置（用户病例）
        self.episodic_config = {
            'host': os.getenv("MILVUS_HOST_EPISODIC", "127.0.0.1"),
            'port': int(os.getenv("MILVUS_PORT_EPISODIC", "19530")),
            'collection': os.getenv("MILVUS_COLLECTION_EPISODIC", "user_cases"),
            'dim': 768
        }
        
        # 语义记忆数据库配置（药品说明书）
        self.semantic_config = {
            'host': os.getenv("MILVUS_HOST_SEMANTIC", "127.0.0.1"),
            'port': int(os.getenv("MILVUS_PORT_SEMANTIC", "19531")),
            'collection': os.getenv("MILVUS_COLLECTION_SEMANTIC", "medical_documents"),
            'dim': 768
        }
        
        # 初始化连接
        self.episodic_client = None
        self.semantic_client = None
        self._init_clients()
    
    def _init_clients(self):
        """初始化向量数据库客户端"""
        try:
            from pymilvus import MilvusClient
            # 情景记忆客户端
            self.episodic_client = MilvusClient(
                uri=f"http://{self.episodic_config['host']}:{self.episodic_config['port']}"
            )
            print(f"✅ 情景记忆数据库连接成功: {self.episodic_config['collection']}")
            
            # 语义记忆客户端
            self.semantic_client = MilvusClient(
                uri=f"http://{self.semantic_config['host']}:{self.semantic_config['port']}"
            )
            print(f"✅ 语义记忆数据库连接成功: {self.semantic_config['collection']}")
            
        except ImportError:
            print("⚠️ 未安装 pymilvus，使用模拟模式")
        except Exception as e:
            print(f"⚠️ 向量数据库连接失败: {e}")
    
    def insert_chunks(self, chunks: List[Dict], store_type: VectorStoreType, user_id: str = ""):
        """
        批量插入文档分块到指定向量数据库
        
        Args:
            chunks: 文档分块列表 [{content, source, ...}]
            store_type: 数据库类型
            user_id: 用户ID（用于情景记忆的权限控制）
        """
        client = self.episodic_client if store_type == VectorStoreType.EPISODIC else self.semantic_client
        collection = self.episodic_config['collection'] if store_type == VectorStoreType.EPISODIC else self.semantic_config['collection']
        
        if not client:
            print(f"⚠️ 模拟插入 {len(chunks)} 个分块到 {store_type.value}")
            return
        
        try:
            data = []
            for chunk in chunks:
                data.append({
                    "content": chunk.get("content", ""),
                    "source": chunk.get("source", chunk.get("filename", "")),
                    "user_id": user_id,
                    "vector": self._encode_query(chunk.get("content", ""))
                })
            
            client.insert(
                collection_name=collection,
                data=data
            )
            print(f"✅ 成功插入 {len(chunks)} 个分块到 {store_type.value}")
        except Exception as e:
            print(f"⚠️ 分块插入失败: {e}")
    
    def search(self, query: str, store_type: VectorStoreType, limit: int = 5) -> List[Dict]:
        """
        在指定向量数据库中检索
        
        Args:
            query: 用户查询
            store_type: 数据库类型（EPISODIC/Semantic）
            limit: 返回结果数量
        
        Returns:
            检索结果列表
        """
        client = self.episodic_client if store_type == VectorStoreType.EPISODIC else self.semantic_client
        collection = self.episodic_config['collection'] if store_type == VectorStoreType.EPISODIC else self.semantic_config['collection']
        
        if not client:
            # 模拟检索结果
            return self._mock_search(query, store_type, limit)
        
        try:
            # 实际向量检索
            results = client.search(
                collection_name=collection,
                data=[self._encode_query(query)],
                limit=limit,
                output_fields=["content", "source", "user_id"]
            )
            return self._format_results(results)
        except Exception as e:
            print(f"⚠️ 检索失败: {e}")
            return self._mock_search(query, store_type, limit)
    
    def search_hybrid(self, query: str, episodic_weight: float = 0.5, limit: int = 5) -> List[Dict]:
        """
        混合检索 - 同时检索情景记忆和语义记忆
        
        Args:
            query: 用户查询
            episodic_weight: 情景记忆权重 (0-1)
            limit: 返回结果数量
        
        Returns:
            融合后的检索结果
        """
        episodic_results = self.search(query, VectorStoreType.EPISODIC, limit)
        semantic_results = self.search(query, VectorStoreType.SEMANTIC, limit)
        
        # 融合结果（基于权重）
        all_results = []
        
        for res in episodic_results:
            res['source_type'] = 'episodic'
            res['score'] = res.get('score', 0) * episodic_weight
            all_results.append(res)
        
        for res in semantic_results:
            res['source_type'] = 'semantic'
            res['score'] = res.get('score', 0) * (1 - episodic_weight)
            all_results.append(res)
        
        # 按分数排序
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return all_results[:limit]
    
    def _encode_query(self, query: str) -> List[float]:
        """将查询编码为向量"""
        global embedding_service
        if embedding_service:
            try:
                embeddings = embedding_service.get_embeddings([query])
                return embeddings[0] if embeddings else [0.0] * self.episodic_config['dim']
            except Exception as e:
                print(f"⚠️ Embedding 服务调用失败: {e}")
        
        # 降级到随机向量
        import random
        return [random.random() for _ in range(self.episodic_config['dim'])]
    
    def _format_results(self, results) -> List[Dict]:
        """格式化检索结果"""
        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    'content': hit.get('entity', {}).get('content', ''),
                    'source': hit.get('entity', {}).get('source', ''),
                    'score': hit.get('distance', 0),
                    'user_id': hit.get('entity', {}).get('user_id', '')
                })
        return formatted
    
    def _mock_search(self, query: str, store_type: VectorStoreType, limit: int) -> List[Dict]:
        """模拟检索结果"""
        if store_type == VectorStoreType.EPISODIC:
            # 情景记忆 - 用户病例
            mock_data = [
                {"content": f"患者病例记录：主诉{query}相关症状，既往病史...", "source": "user_case_001", "score": 0.92},
                {"content": f"就诊记录：患者曾因{query}症状就诊，诊断结果...", "source": "user_case_002", "score": 0.87},
                {"content": f"检查报告：{query}相关检查结果分析...", "source": "user_case_003", "score": 0.81},
            ]
        else:
            # 语义记忆 - 药品说明书
            mock_data = [
                {"content": f"药品说明书：{query}相关药物的适应症、用法用量...", "source": "drug_manual_001", "score": 0.95},
                {"content": f"医学文献：{query}相关治疗指南...", "source": "medical_literature_001", "score": 0.89},
                {"content": f"药品说明：{query}药物相互作用注意事项...", "source": "drug_manual_002", "score": 0.83},
            ]
        
        return mock_data[:limit]
    
    def insert_document(self, content: str, source: str, store_type: VectorStoreType, user_id: str = ""):
        """
        插入文档到指定向量数据库
        
        Args:
            content: 文档内容
            source: 文档来源标识
            store_type: 数据库类型
            user_id: 用户ID（用于情景记忆的权限控制）
        """
        client = self.episodic_client if store_type == VectorStoreType.EPISODIC else self.semantic_client
        collection = self.episodic_config['collection'] if store_type == VectorStoreType.EPISODIC else self.semantic_config['collection']
        
        if not client:
            print(f"⚠️ 模拟插入到 {store_type.value}: {source}")
            return
        
        try:
            client.insert(
                collection_name=collection,
                data=[{
                    "content": content,
                    "source": source,
                    "user_id": user_id,
                    "vector": self._encode_query(content)
                }]
            )
            print(f"✅ 文档插入成功: {source}")
        except Exception as e:
            print(f"⚠️ 文档插入失败: {e}")

# 全局实例
_dual_vector_store = None

def get_dual_vector_store() -> DualVectorStore:
    """获取双向量数据库实例"""
    global _dual_vector_store
    if _dual_vector_store is None:
        _dual_vector_store = DualVectorStore()
    return _dual_vector_store

# 便捷函数
def search_episodic_memory(query: str, limit: int = 5) -> List[Dict]:
    """检索情景记忆（用户病例）"""
    return get_dual_vector_store().search(query, VectorStoreType.EPISODIC, limit)

def search_semantic_memory(query: str, limit: int = 5) -> List[Dict]:
    """检索语义记忆（药品说明书）"""
    return get_dual_vector_store().search(query, VectorStoreType.SEMANTIC, limit)

def search_hybrid_memory(query: str, episodic_weight: float = 0.5, limit: int = 5) -> List[Dict]:
    """混合检索"""
    return get_dual_vector_store().search_hybrid(query, episodic_weight, limit)

def insert_episodic_memory(chunks: List[Dict], user_id: str = ""):
    """插入到情景记忆（用户病例）"""
    get_dual_vector_store().insert_chunks(chunks, VectorStoreType.EPISODIC, user_id)

def insert_semantic_memory(chunks: List[Dict], user_id: str = ""):
    """插入到语义记忆（药品说明书）"""
    get_dual_vector_store().insert_chunks(chunks, VectorStoreType.SEMANTIC, user_id)

if __name__ == "__main__":
    # 测试
    store = get_dual_vector_store()
    
    print("\n=== 测试情景记忆检索 ===")
    results = store.search("糖尿病", VectorStoreType.EPISODIC)
    for i, res in enumerate(results):
        print(f"{i+1}. {res['source']} (score: {res['score']:.2f})")
    
    print("\n=== 测试语义记忆检索 ===")
    results = store.search("阿莫西林", VectorStoreType.SEMANTIC)
    for i, res in enumerate(results):
        print(f"{i+1}. {res['source']} (score: {res['score']:.2f})")
    
    print("\n=== 测试混合检索 ===")
    results = store.search_hybrid("糖尿病用药")
    for i, res in enumerate(results):
        print(f"{i+1}. [{res['source_type']}] {res['source']} (score: {res['score']:.2f})")
