from typing import Optional
import os
import requests
from datetime import datetime, timezone
import time
from dotenv import load_dotenv
try:
    from langchain_core.tools import tool
except ImportError:
    from langchain_core.tools import tool

load_dotenv()

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

_LAST_RAG_CONTEXT = None
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_RAG_STEP_QUEUE = None  # asyncio.Queue, set by agent before streaming
_RAG_STEP_LOOP = None   # asyncio loop, captured when setting queue
_RAG_STEP_START_MONO = None
_RAG_STEP_LAST_MONO = None
_RAG_STEP_INDEX = 0


def _set_last_rag_context(context: dict):
    global _LAST_RAG_CONTEXT
    _LAST_RAG_CONTEXT = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    global _LAST_RAG_CONTEXT
    context = _LAST_RAG_CONTEXT
    if clear:
        _LAST_RAG_CONTEXT = None
    return context


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP, _RAG_STEP_START_MONO, _RAG_STEP_LAST_MONO, _RAG_STEP_INDEX
    _RAG_STEP_QUEUE = queue
    if queue:
        import asyncio
        try:
            _RAG_STEP_LOOP = asyncio.get_running_loop()
        except RuntimeError:
            _RAG_STEP_LOOP = asyncio.get_event_loop()
        _RAG_STEP_START_MONO = time.perf_counter()
        _RAG_STEP_LAST_MONO = _RAG_STEP_START_MONO
        _RAG_STEP_INDEX = 0
    else:
        _RAG_STEP_LOOP = None
        _RAG_STEP_START_MONO = None
        _RAG_STEP_LAST_MONO = None
        _RAG_STEP_INDEX = 0


def emit_rag_step(icon: str, label: str, detail: str = "", extra: Optional[dict] = None):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP, _RAG_STEP_START_MONO, _RAG_STEP_LAST_MONO, _RAG_STEP_INDEX
    if _RAG_STEP_QUEUE is not None and _RAG_STEP_LOOP is not None:
        now_mono = time.perf_counter()
        if _RAG_STEP_START_MONO is None:
            _RAG_STEP_START_MONO = now_mono
        if _RAG_STEP_LAST_MONO is None:
            _RAG_STEP_LAST_MONO = now_mono
        _RAG_STEP_INDEX += 1
        step = {
            "icon": icon,
            "label": label,
            "detail": detail,
            "step_index": _RAG_STEP_INDEX,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((now_mono - _RAG_STEP_START_MONO) * 1000),
            "delta_ms": round((now_mono - _RAG_STEP_LAST_MONO) * 1000),
        }
        if extra:
            step.update(extra)
        _RAG_STEP_LAST_MONO = now_mono
        try:
            if not _RAG_STEP_LOOP.is_closed():
                _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, step)
        except Exception:
            pass


def emit_rag_payload(payload: dict):
    """发送任意 RAG 事件负载，适合传输检索文档、原始 JSON 等结构化数据。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    if _RAG_STEP_QUEUE is not None and _RAG_STEP_LOOP is not None:
        try:
            if not _RAG_STEP_LOOP.is_closed():
                _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, payload)
        except Exception:
            pass


def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """获取天气信息"""
    if not location:
        return "location参数不能为空"
    if extensions not in ("base", "all"):
        return "extensions参数错误，请输入base或all"

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）"

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return f"查询失败：{data.get('info', '未知错误')}"

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return f"未查询到 {location} 的天气数据"
            w = lives[0]
            return (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            )

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"未查询到 {location} 的天气预报数据"
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "今日天气：",
            f"  白天：{today.get('dayweather','未知')}",
            f"  夜间：{today.get('nightweather','未知')}",
            f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
        ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "错误：请求天气服务超时"
    except requests.exceptions.RequestException as e:
        return f"错误：天气服务请求失败 - {e}"
    except Exception as e:
        return f"错误：解析天气数据失败 - {e}"


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    # ... guards omitted ...
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1

    from .rag_pipeline import run_rag_graph

    # 在同步工具中获取当前的 Loop 可能不可靠，但我们之前是通过 call_soon_threadsafe 调度的。
    # 这里 _RAG_STEP_QUEUE 是在主线程/Loop 设置的全局变量。
    # 如果工具运行在线程池中，它是可以访问到全局变量 _RAG_STEP_QUEUE 的。
    # emit_rag_step 内部做了 try-except 和 get_event_loop()。

    # 问题可能出在 asyncio.get_event_loop() 在子线程中调用会报错或者拿不到主线程的loop。
    # 我们应该在 set_rag_step_queue 时也保存 loop 引用，或者在 emit_rag_step 中更健壮地获取 loop。

    rag_result = run_rag_graph(query)

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        _set_last_rag_context({"rag_trace": rag_trace})

    if not docs:
        return "No relevant documents found in the knowledge base."

    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)
# ───────────────────────────────────────────
# Neo4j 医疗知识图谱工具
# ───────────────────────────────────────────
import os
from typing import Optional
 
_neo4j_graph = None
_neo4j_last_error = None
 
def _get_neo4j_graph():
    """懒加载 Neo4j 连接，失败时返回 None（降级）。"""
    global _neo4j_graph, _neo4j_last_error
    if _neo4j_graph is not None:
        return _neo4j_graph
    try:
        from py2neo import Graph
        user = os.getenv("NEO4J_USER", "neo4j")
        pwd  = os.getenv("NEO4J_PASSWORD", "supermew123")
        db   = os.getenv("NEO4J_DBNAME", "neo4j")
        preferred = os.getenv("NEO4J_URL", "bolt://127.0.0.1:7687")
        candidates = [
            preferred,
            "bolt://127.0.0.1:7687",
            "http://127.0.0.1:7474",
        ]
        seen = set()
        deduped = []
        for url in candidates:
            if url and url not in seen:
                seen.add(url)
                deduped.append(url)
        for url in deduped:
            try:
                g = Graph(url, auth=(user, pwd), name=db)
                g.run("RETURN 1 AS ok").data()
                _neo4j_graph = g
                _neo4j_last_error = None
                return _neo4j_graph
            except Exception as inner_exc:
                _neo4j_last_error = str(inner_exc)
                continue
    except Exception as e:
        _neo4j_last_error = str(e)
        return None
    return None
 
 
# 意图 → Cypher 模板映射表
# 每条 Cypher 中 {entity} 会被实际实体名替换
_INTENT_CYPHER = {
    "疾病简介":   "MATCH (a:疾病{{名称:'{entity}'}}) RETURN a.疾病简介 AS result",
    "疾病病因":   "MATCH (a:疾病{{名称:'{entity}'}}) RETURN a.疾病病因 AS result",
    "预防措施":   "MATCH (a:疾病{{名称:'{entity}'}}) RETURN a.预防措施 AS result",
    "治疗周期":   "MATCH (a:疾病{{名称:'{entity}'}}) RETURN a.治疗周期 AS result",
    "治愈概率":   "MATCH (a:疾病{{名称:'{entity}'}}) RETURN a.治愈概率 AS result",
    "易感人群":   "MATCH (a:疾病{{名称:'{entity}'}}) RETURN a.疾病易感人群 AS result",
    "推荐药品":   "MATCH (a:疾病{{名称:'{entity}'}})-[:疾病使用药品]->(b:药品) RETURN b.名称 AS result",
    "常用药品":   "MATCH (a:疾病{{名称:'{entity}'}})-[:疾病使用药品]->(b:药品) RETURN b.名称 AS result",
    "宜吃食物":   "MATCH (a:疾病{{名称:'{entity}'}})-[:疾病宜吃食物]->(b:食物) RETURN b.名称 AS result",
    "忌吃食物":   "MATCH (a:疾病{{名称:'{entity}'}})-[:疾病忌吃食物]->(b:食物) RETURN b.名称 AS result",
    "所需检查":   "MATCH (a:疾病{{名称:'{entity}'}})-[:疾病所需检查]->(b:检查项目) RETURN b.名称 AS result",
    "所属科室":   "MATCH (a:疾病{{名称:'{entity}'}})-[:疾病所属科目]->(b:科目) RETURN b.名称 AS result",
    "疾病症状":   "MATCH (a:疾病{{名称:'{entity}'}})-[:疾病的症状]->(b:疾病症状) RETURN b.名称 AS result",
    "治疗方法":   "MATCH (a:疾病{{名称:'{entity}'}})-[:治疗的方法]->(b:治疗方法) RETURN b.名称 AS result",
    "并发疾病":   "MATCH (a:疾病{{名称:'{entity}'}})-[:疾病并发疾病]->(b:疾病) RETURN b.名称 AS result",
    "药品生产商": "MATCH (a:药品商)-[:生产]->(b:药品{{名称:'{entity}'}}) RETURN a.名称 AS result",
}

_INTENT_RELATION_LABEL = {
    "疾病简介": "疾病属性: 疾病简介",
    "疾病病因": "疾病属性: 疾病病因",
    "预防措施": "疾病属性: 预防措施",
    "治疗周期": "疾病属性: 治疗周期",
    "治愈概率": "疾病属性: 治愈概率",
    "易感人群": "疾病属性: 疾病易感人群",
    "推荐药品": "疾病使用药品",
    "常用药品": "疾病使用药品",
    "宜吃食物": "疾病宜吃食物",
    "忌吃食物": "疾病忌吃食物",
    "所需检查": "疾病所需检查",
    "所属科室": "疾病所属科目",
    "疾病症状": "疾病的症状",
    "治疗方法": "治疗的方法",
    "并发疾病": "疾病并发疾病",
    "药品生产商": "生产",
}


def _run_cypher(graph, cypher: str):
    try:
        return graph.run(cypher).data()
    except Exception:
        return []


def _candidate_entities(graph, entity: str):
    candidates = []
    seen = set()
    queries = [
        f"MATCH (d:疾病) WHERE d.名称 = '{entity}' RETURN d.名称 AS name LIMIT 10",
        f"MATCH (d:疾病) WHERE d.名称 CONTAINS '{entity}' RETURN d.名称 AS name LIMIT 10",
        f"MATCH (d:疾病)-[:疾病的症状]->(s:疾病症状) WHERE s.名称 CONTAINS '{entity}' RETURN DISTINCT d.名称 AS name LIMIT 10",
        f"MATCH (d:疾病)-[:疾病所需检查]->(c:检查项目) WHERE c.名称 CONTAINS '{entity}' RETURN DISTINCT d.名称 AS name LIMIT 10",
    ]
    for query in queries:
        for row in _run_cypher(graph, query):
            name = row.get("name")
            if name and name not in seen:
                seen.add(name)
                candidates.append(name)
    return candidates[:10]


def _format_multi_result(entity: str, intent: str, results: list[dict], matched_entity: str | None = None) -> str:
    values = [str(r.get("result", "")).strip() for r in results if str(r.get("result", "")).strip()]
    if not values:
        return ""
    prefix = f"【{matched_entity or entity}】{intent}"
    if len(values) == 1:
        return f"{prefix}：\n{values[0]}"
    return f"{prefix}（共{len(values)}项）：\n" + "、".join(values)
 
 
@tool("search_medical_graph")
def search_medical_graph(entity: str, intent: str) -> str:
    """
    Query the medical knowledge graph (Neo4j) for structured medical facts.
    Use this tool for precise factual questions about diseases, drugs, symptoms, or treatments.
 
    Args:
        entity: The medical entity to query, e.g. '糖尿病', '阿司匹林'
        intent: The type of information needed. Must be one of:
                疾病简介 | 疾病病因 | 预防措施 | 治疗周期 | 治愈概率 | 易感人群 |
                推荐药品 | 常用药品 | 宜吃食物 | 忌吃食物 | 所需检查 |
                所属科室 | 疾病症状 | 治疗方法 | 并发疾病 | 药品生产商
 
    Examples:
        entity='糖尿病', intent='忌吃食物'
        entity='高血压', intent='推荐药品'
        entity='感冒',   intent='疾病症状'
    """
    graph = _get_neo4j_graph()
    if graph is None:
        emit_rag_step("⚠️", "图谱连接失败", "本轮使用知识库语义检索补充")
        return "知识图谱本轮未返回可用结果。"
 
    cypher_template = _INTENT_CYPHER.get(intent)
    if not cypher_template:
        available = " | ".join(_INTENT_CYPHER.keys())
        return f"不支持的意图类型：'{intent}'。支持的类型：{available}"
 
    candidate_entities = _candidate_entities(graph, entity)
    if not candidate_entities:
        candidate_entities = [entity]

    emit_rag_step(
        "🔎",
        "命中实体",
        f"输入: {entity}",
        extra={"layer": "entity", "items": candidate_entities[:10], "matched_input": entity},
    )

    relation_label = _INTENT_RELATION_LABEL.get(intent, intent)
    emit_rag_step(
        "🔗",
        "命中关系",
        f"意图: {intent}",
        extra={"layer": "relation", "relation": relation_label, "intent": intent},
    )

    emit_rag_step("🧭", "图谱候选实体", f"{len(candidate_entities)} 个候选")

    all_records = []
    matched_entities = []
    evidence_sources = []
    for candidate in candidate_entities:
        cypher = cypher_template.format(entity=candidate)
        records = _run_cypher(graph, cypher)
        if records:
            all_records.extend(records)
            matched_entities.append(candidate)
            evidence_sources.append({"entity": candidate, "records": records[:10]})

    if not all_records:
        emit_rag_step("⚠️", "图谱无结果", f"未找到 {entity} 的 {intent} 信息")
        return (
            f"知识图谱中未找到关于「{entity}」的「{intent}」信息。"
            f"可尝试更具体的关键词，或改用 search_knowledge_base 进行语义检索。"
        )

    deduped = []
    seen = set()
    for row in all_records:
        value = str(row.get("result", "")).strip()
        if value and value not in seen:
            seen.add(value)
            deduped.append({"result": value})

    matched_label = matched_entities[0] if matched_entities else entity
    emit_rag_step(
        "📄",
        "命中证据",
        f"{len(deduped)} 条证据",
        extra={
            "layer": "evidence",
            "matched_entity": matched_label,
            "hits": deduped[:10],
            "source_entities": matched_entities[:10],
            "evidence_sources": evidence_sources[:5],
        },
    )
    emit_rag_step("✅", "图谱命中", f"{matched_label} · {intent} · {len(deduped)}条")
    if len(matched_entities) > 1:
        emit_rag_step("ℹ️", "关键词扩展命中", f"匹配实体: {'、'.join(matched_entities[:5])}")

    return _format_multi_result(entity, intent, deduped, matched_entity=matched_label)