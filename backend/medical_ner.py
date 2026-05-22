"""医疗实体识别工具 - 使用 HuggingFace Transformers"""
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import os
import re

# 医疗实体类型映射
ENTITY_TYPES = {
    'DISEASE': '疾病',
    'SYMPTOM': '症状',
    'DRUG': '药品',
    'TREATMENT': '治疗方法',
    'CHECK': '检查项目',
}

class MedicalNER:
    """医疗实体识别类"""
    
    def __init__(self):
        self.ner = None
        self._load_model()
    
    def _load_model(self):
        """加载 NER 模型"""
        try:
            # 医疗领域专用 NER 模型列表
            # 优先使用经过医疗领域微调的模型
            medical_models = [
                "dmis-lab/biobert-v1.1",           # 生物医学领域专用 BERT
                "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",  # 医疗语义匹配模型
                "zhihan1996/Chinese-BERT-wwm-ext-for-NER",  # 中文 NER 专用
            ]
            
            # 通用中文模型作为备选
            fallback_models = [
                "hfl/chinese-roberta-wwm-ext",
                "bert-base-chinese"
            ]
            
            for model in medical_models + fallback_models:
                try:
                    self.ner = pipeline(
                        "ner",
                        model=model,
                        tokenizer=model,
                        aggregation_strategy="simple",
                        trust_remote_code=True
                    )
                    print(f"✅ 成功加载 NER 模型: {model}")
                    return
                except Exception as e:
                    print(f"⚠️ 加载模型 {model} 失败: {e}")
                    continue
            
            # 如果都失败，使用基于规则的识别
            print("⚠️ 未加载到预训练模型，使用规则匹配")
            self.ner = None
            
        except ImportError:
            print("⚠️ 未安装 transformers 库，使用规则匹配")
            self.ner = None
    
    def extract_entities(self, text: str) -> list:
        """
        从文本中提取医疗实体
        
        Args:
            text: 用户输入文本
        
        Returns:
            实体列表，每个元素包含 entity_type, entity, score
        """
        if self.ner:
            return self._extract_with_model(text)
        else:
            return self._extract_with_rules(text)
    
    def _extract_with_model(self, text: str) -> list:
        """使用预训练模型提取实体"""
        try:
            results = self.ner(text)
            entities = []
            for result in results:
                entities.append({
                    'entity_type': result['entity_group'],
                    'entity': result['word'],
                    'score': round(result['score'], 4)
                })
            return entities
        except Exception as e:
            print(f"⚠️ NER 模型调用失败: {e}")
            return self._extract_with_rules(text)
    
    def _extract_with_rules(self, text: str) -> list:
        """使用规则匹配提取实体（降级方案）"""
        entities = []
        seen = set()
        
        # 医疗实体词典 - 扩展版
        diseases = [
            '糖尿病', '高血压', '牙龈炎', '牙周炎', '感冒', '发烧', '咳嗽', 
            '头痛', '胃痛', '关节炎', '心脏病', '肺炎', '肝炎', '癌症',
            '胃炎', '肠炎', '肾炎', '脑炎', '脑膜炎', '结膜炎', '中耳炎',
            '鼻窦炎', '支气管炎', '肺炎', '哮喘', '肺结核', '肺气肿',
            '冠心病', '心律失常', '心力衰竭', '心肌梗死', '中风', '血栓',
            '糖尿病', '低血糖', '甲亢', '甲减', '痛风', '类风湿关节炎',
            '骨质疏松', '贫血', '白血病', '淋巴瘤', '肝硬化', '胆结石',
            '肾结石', '肾炎', '尿路感染', '前列腺炎', '宫颈癌', '乳腺癌',
            '肺癌', '胃癌', '肝癌', '肠癌', '胰腺癌', '卵巢癌', '子宫内膜癌',
            '抑郁症', '焦虑症', '失眠症', '癫痫', '帕金森', '阿尔茨海默病',
        ]
        
        symptoms = [
            '肿', '痛', '痒', '酸', '胀', '麻', '晕', '吐', '泻', 
            '发热', '乏力', '失眠', '食欲不振', '呼吸困难',
            '头痛', '头晕', '恶心', '呕吐', '腹泻', '便秘',
            '咳嗽', '咳痰', '胸闷', '胸痛', '心悸', '气短',
            '口干', '口苦', '口臭', '牙龈出血', '牙痛', '咽喉痛',
            '关节痛', '肌肉痛', '腰痛', '腹痛', '胃痛', '痛经',
            '视力模糊', '耳鸣', '听力下降', '皮肤瘙痒', '皮疹', '红斑',
            '水肿', '黄疸', '消瘦', '肥胖', '多汗', '盗汗',
        ]
        
        drugs = [
            '阿司匹林', '青霉素', '布洛芬', '维生素', '中药', '西药',
            '阿莫西林', '头孢', '红霉素', '四环素', '氯霉素', '磺胺',
            '奥美拉唑', '兰索拉唑', '雷贝拉唑', '泮托拉唑',
            '硝苯地平', '氨氯地平', '缬沙坦', '氯沙坦', '卡托普利',
            '胰岛素', '二甲双胍', '格列齐特', '阿卡波糖',
            '布洛芬', '对乙酰氨基酚', '萘普生', '双氯芬酸钠',
            '氨溴索', '沙丁胺醇', '布地奈德', '异丙托溴铵',
            '蒙脱石', '双歧杆菌', '乳果糖', '聚乙二醇',
        ]
        
        treatments = [
            '手术', '化疗', '放疗', '针灸', '推拿', '按摩',
            '理疗', '热敷', '冷敷', '牵引', '拔罐', '刮痧',
            '输液', '输血', '透析', '雾化', '吸氧',
        ]
        
        checks = [
            '血常规', '尿常规', '肝功能', '肾功能', '心电图', 'B超',
            'CT', 'MRI', 'X光', '胃镜', '肠镜', '血糖',
            '血压', '血脂', '甲状腺功能', '肿瘤标志物', '核酸检测',
        ]
        
        # 匹配疾病
        for disease in diseases:
            if disease in text and disease not in seen:
                seen.add(disease)
                entities.append({
                    'entity_type': 'DISEASE',
                    'entity': disease,
                    'score': 0.9
                })
        
        # 匹配症状
        for symptom in symptoms:
            if symptom in text and symptom not in seen:
                seen.add(symptom)
                entities.append({
                    'entity_type': 'SYMPTOM',
                    'entity': symptom,
                    'score': 0.85
                })
        
        # 匹配药品
        for drug in drugs:
            if drug in text and drug not in seen:
                seen.add(drug)
                entities.append({
                    'entity_type': 'DRUG',
                    'entity': drug,
                    'score': 0.9
                })
        
        # 匹配治疗方法
        for treatment in treatments:
            if treatment in text and treatment not in seen:
                seen.add(treatment)
                entities.append({
                    'entity_type': 'TREATMENT',
                    'entity': treatment,
                    'score': 0.85
                })
        
        # 匹配检查项目
        for check in checks:
            if check in text and check not in seen:
                seen.add(check)
                entities.append({
                    'entity_type': 'CHECK',
                    'entity': check,
                    'score': 0.9
                })
        
        # 使用正则匹配疾病名称模式
        disease_patterns = [
            (r'(\w+炎)', 0.8),    # 如：牙龈炎、肺炎
            (r'(\w+病)', 0.8),    # 如：糖尿病、心脏病
            (r'(\w+症)', 0.75),   # 如：高血压症、炎症
            (r'(\w+癌)', 0.9),    # 如：肺癌、胃癌
            (r'(\w+瘤)', 0.85),   # 如：肿瘤、肌瘤
            (r'(\w+毒)', 0.7),    # 如：病毒、中毒
        ]
        
        for pattern, score in disease_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2 and match not in seen:  # 至少两个字符
                    seen.add(match)
                    entities.append({
                        'entity_type': 'DISEASE',
                        'entity': match,
                        'score': score
                    })
        
        # 使用正则匹配症状模式
        symptom_patterns = [
            (r'(\w+痛)', 0.85),   # 如：头痛、胃痛
            (r'(\w+胀)', 0.8),    # 如：胃胀、腹胀
            (r'(\w+麻)', 0.8),    # 如：麻木
            (r'(\w+晕)', 0.8),    # 如：头晕
            (r'(\w+吐)', 0.8),    # 如：呕吐
            (r'(\w+泻)', 0.8),    # 如：腹泻
            (r'(\w+烧)', 0.8),    # 如：发烧
            (r'(\w+痒)', 0.85),   # 如：瘙痒
        ]
        
        for pattern, score in symptom_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= 2 and match not in seen:
                    seen.add(match)
                    entities.append({
                        'entity_type': 'SYMPTOM',
                        'entity': match,
                        'score': score
                    })
        
        return entities

# 全局实例
_medical_ner = None

def get_medical_ner() -> MedicalNER:
    """获取医疗 NER 实例（单例模式）"""
    global _medical_ner
    if _medical_ner is None:
        _medical_ner = MedicalNER()
    return _medical_ner

def extract_medical_entities(text: str) -> list:
    """便捷函数：提取医疗实体"""
    return get_medical_ner().extract_entities(text)

if __name__ == "__main__":
    # 测试
    test_texts = [
        "我的牙龈肿了怎么办",
        "糖尿病患者应该注意什么",
        "感冒发烧吃什么药",
    ]
    
    ner = get_medical_ner()
    for text in test_texts:
        entities = ner.extract_entities(text)
        print(f"\n文本: {text}")
        print(f"实体: {entities}")