from dotenv import load_dotenv
import os
import json
import asyncio
import re
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage, message_to_dict, messages_from_dict
from .tools import (
    get_current_weather, search_knowledge_base, search_medical_graph,
    get_last_rag_context, reset_tool_call_guards, set_rag_step_queue,
    emit_rag_step
)
from .intelligent_retrieval import get_intelligent_retrieval
from datetime import datetime
from .cache import cache
from .database import SessionLocal
from .models import User, ChatSession, ChatMessage

load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

class ConversationStorage:
    """对话存储（PostgreSQL + Redis）。"""

    @staticmethod
    def _messages_cache_key(user_id: str, session_id: str) -> str:
        return f"chat_messages:{user_id}:{session_id}"

    @staticmethod
    def _sessions_cache_key(user_id: str) -> str:
        return f"chat_sessions:{user_id}"

    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        messages = []
        for msg_data in records:
            message_dict = msg_data.get("message_dict") if isinstance(msg_data, dict) else None
            if message_dict:
                try:
                    messages.append(messages_from_dict([message_dict])[0])
                    continue
                except Exception:
                    pass
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content, additional_kwargs=msg_data.get("additional_kwargs", {})))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        return messages

    def save(self, user_id: str, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None):
        """保存对话"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return

            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                session = ChatSession(user_id=user.id, session_id=session_id, metadata_json=metadata or {})
                db.add(session)
                db.flush()
            else:
                session.metadata_json = metadata or {}

            db.query(ChatMessage).filter(ChatMessage.session_ref_id == session.id).delete(synchronize_session=False)

            serialized = []
            now = datetime.utcnow()
            for idx, msg in enumerate(messages):
                rag_trace = None
                message_meta = message_to_dict(msg)
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    rag_trace = extra.get("rag_trace")
                    message_meta["data"]["additional_kwargs"] = {
                        **message_meta.get("data", {}).get("additional_kwargs", {}),
                        **extra.get("additional_kwargs", {}),
                    }
                    response_metadata = extra.get("response_metadata")
                    if response_metadata:
                        message_meta["data"]["response_metadata"] = response_metadata

                db.add(
                    ChatMessage(
                        session_ref_id=session.id,
                        message_type=msg.type,
                        content=str(msg.content),
                        timestamp=now,
                        rag_trace={"rag_trace": rag_trace, "message_dict": message_meta},
                    )
                )
                serialized.append(
                    {
                        "type": msg.type,
                        "content": str(msg.content),
                        "timestamp": now.isoformat(),
                        "rag_trace": rag_trace,
                        "message_dict": message_meta,
                    }
                )

            session.updated_at = now
            db.commit()

            cache.set_json(self._messages_cache_key(user_id, session_id), serialized)
            cache.delete(self._sessions_cache_key(user_id))
        finally:
            db.close()

    def load(self, user_id: str, session_id: str) -> list:
        """加载对话"""
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return self._to_langchain_messages(cached)

        records = self.get_session_messages(user_id, session_id)
        cache.set_json(self._messages_cache_key(user_id, session_id), records)
        return self._to_langchain_messages(records)

    def list_sessions(self, user_id: str) -> list:
        """列出用户的所有会话"""
        return [item["session_id"] for item in self.list_session_infos(user_id)]

    def list_session_infos(self, user_id: str) -> list[dict]:
        cached = cache.get_json(self._sessions_cache_key(user_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []

            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )
            result = []
            for s in sessions:
                count = db.query(ChatMessage).filter(ChatMessage.session_ref_id == s.id).count()
                result.append(
                    {
                        "session_id": s.session_id,
                        "updated_at": s.updated_at.isoformat(),
                        "message_count": count,
                    }
                )
            cache.set_json(self._sessions_cache_key(user_id), result)
            return result
        finally:
            db.close()

    def get_session_messages(self, user_id: str, session_id: str) -> list[dict]:
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return []

            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            result = [
                {
                    "type": row.message_type,
                    "content": row.content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": row.rag_trace.get("rag_trace") if isinstance(row.rag_trace, dict) else row.rag_trace,
                    "message_dict": row.rag_trace.get("message_dict") if isinstance(row.rag_trace, dict) else None,
                }
                for row in rows
            ]
            cache.set_json(self._messages_cache_key(user_id, session_id), result)
            return result
        finally:
            db.close()

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除指定用户的会话，返回是否删除成功"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return False
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return False

            db.delete(session)
            db.commit()
            cache.delete(self._messages_cache_key(user_id, session_id))
            cache.delete(self._sessions_cache_key(user_id))
            return True
        finally:
            db.close()


def create_agent_instance():
    model = init_chat_model(
        model=MODEL,
        model_provider="deepseek",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.3,
        stream_usage=True,
        extra_body={"thinking": {"type": "disabled"}},
    )

    agent = create_agent(
        model=model,
        tools=[get_current_weather, search_knowledge_base, search_medical_graph],  # ← 加这个
        system_prompt=(
            "You are a helpful assistant. You have access to three tools:\n\n"

            "1. search_medical_graph — Query a structured medical knowledge graph.\n"
            "   Use this for PRECISE factual questions like:\n"
            "   - What foods should a diabetic avoid? (entity='糖尿病', intent='忌吃食物')\n"
            "   - What are the symptoms of hypertension? (entity='高血压', intent='疾病症状')\n"
            "   - What drugs are recommended for pneumonia? (entity='肺炎', intent='推荐药品')\n"
            "   Trigger words: 症状、药品、忌吃、宜吃、检查、治疗方法、并发、科室、病因、预防\n\n"

            "2. search_knowledge_base — Semantic search over uploaded documents.\n"
            "   Use this for OPEN-ENDED questions, explanations, or when the topic\n"
            "   is unlikely to be in a structured graph (e.g., recent research, policies).\n\n"

            "3. get_current_weather — Get real-time weather for a location.\n\n"

            "ROUTING RULES:\n"
            "- Structured medical fact → search_medical_graph FIRST.\n"
            "  If it returns no result, fall back to search_knowledge_base.\n"
            "- Open-ended medical question → search_knowledge_base.\n"
            "- Non-medical question → other tools or answer directly.\n\n"

            "LIMITS:\n"
            "- Call search_knowledge_base at most ONCE per turn.\n"
            "- After receiving any tool result, produce the Final Answer immediately.\n"
            "- Never fabricate medical facts. If unknown, say so."
        ),
    )
    return agent, model


agent, model = create_agent_instance()

storage = ConversationStorage()


def _message_content_to_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = ""
        for block in content:
            if isinstance(block, str):
                out += block
            elif isinstance(block, dict) and block.get("type") == "text":
                out += block.get("text", "")
        return out
    return str(content)


_MEDICAL_ENTITY_HINTS = [
    "腹泻", "拉肚子", "发烧", "发热", "喉咙痛", "咽痛", "咳嗽", "头痛", "腹痛", "呕吐", "恶心", "海鲜",
    "糖尿病", "高血压", "感冒", "肠炎", "胃炎", "过敏", "皮疹", "便秘", "胸痛", "牙痛", "牙齿痛",
]

_MEDICAL_INTENT_KEYWORDS = (
    "疾病", "症状", "病因", "预防", "治疗", "用药", "药", "检查", "检验", "化验", "并发", "科室",
    "忌口", "宜吃", "忌吃", "能不能吃", "可以吃", "不舒服", "疼", "痛",
)

_KNOWLEDGE_BASE_KEYWORDS = (
    "知识库", "文档", "资料", "检索", "根据上传", "基于", "总结", "说明", "解释", "流程", "实现",
    "代码", "项目", "论文", "报告",
)


def _is_medical_query(user_text: str) -> bool:
    text = user_text or ""
    return any(term in text for term in _MEDICAL_ENTITY_HINTS) or any(
        keyword in text for keyword in _MEDICAL_INTENT_KEYWORDS
    )


def _should_prefetch_knowledge_base(user_text: str, graph_outputs: list[str]) -> bool:
    return True


def _extract_medical_entity(user_text: str) -> str:
    text = (user_text or "").strip()
    for term in sorted(_MEDICAL_ENTITY_HINTS, key=len, reverse=True):
        if term in text:
            return term
    lead = re.split(r"[，。！？、,;；\s]", text, maxsplit=1)[0].strip()
    return lead or text[:8]


def _select_graph_intents(user_text: str) -> list[str]:
    text = user_text or ""
    intents = []
    keyword_map = [
        (r"因为|原因|导致|为什么|是因为", ["疾病病因"]),
        (r"症状|表现", ["疾病症状"]),
        (r"注意|怎么办|护理|处理|治疗", ["预防措施", "治疗方法", "所需检查"]),
        (r"忌吃|忌口|不能吃|少吃|避免吃", ["忌吃食物"]),
        (r"宜吃|适合吃|推荐吃", ["宜吃食物"]),
        (r"能不能吃|可以吃", ["宜吃食物", "忌吃食物"]),
        (r"药|用药|止泻|退烧", ["推荐药品", "常用药品"]),
        (r"检查|检验|化验", ["所需检查"]),
    ]
    for pattern, mapped in keyword_map:
        if re.search(pattern, text):
            for item in mapped:
                if item not in intents:
                    intents.append(item)
    if re.search(r"是什么|介绍|简介|了解", text) and not re.search(r"因为|原因|导致|为什么|是因为", text):
        intents.append("疾病简介")
    if not intents:
        intents.append("疾病简介")
    return intents[:4]


def _prefetch_medical_context(user_text: str) -> str:
    """Run deterministic pre-retrieval only when it improves routing clarity."""
    # 使用智能检索器进行检索
    retriever = get_intelligent_retrieval()
    
    try:
        # 执行智能检索
        retrieval_result = retriever.retrieve(user_text)
        
        # 检查是否有检索结果
        results = retrieval_result['results']
        has_results = any(len(items) > 0 for items in results.values())
        
        if not has_results:
            return ""
        
        # 格式化检索结果
        formatted_results = retriever.format_results(retrieval_result)
        
        # 构建最终上下文
        return (
            "你已经获得以下预检索上下文，这些内容已经是工具检索结果。请只基于这些内容回答，并在必要时说明不确定性。\n"
            "检索策略: " + retrieval_result['strategy'] + "\n"
            "问题类型: " + retrieval_result['question_type'] + "\n\n"
            "如果上下文包含“医疗知识图谱”章节，说明知识图谱本轮已经可用且已返回证据，禁止说“知识图谱不可用/暂时不可用”。\n"
            "如果检索结果与用户问题主题不匹配，请明确说明知识库未检索到相关资料，不要用通用知识补全成确定答案。\n"
            "不要再次调用任何工具，直接基于上下文作答。\n\n"
            + formatted_results
        )
    
    except Exception as exc:
        emit_rag_step('❌', '智能检索失败', str(exc))
        # 降级到原有逻辑
        return _prefetch_medical_context_fallback(user_text)


def _prefetch_medical_context_fallback(user_text: str) -> str:
    """降级检索逻辑 - 当智能检索失败时使用"""
    sections: list[str] = []
    graph_outputs: list[str] = []

    if _is_medical_query(user_text):
        entity = _extract_medical_entity(user_text)
        intents = _select_graph_intents(user_text)
        
        if entity and intents:
            emit_rag_step('🧬', '开始知识图谱检索', f'识别实体: {entity}')
        
        for intent in intents:
            emit_rag_step('🔍', '查询知识图谱', f'{entity} → {intent}')
            try:
                graph_output = search_medical_graph.invoke({"entity": entity, "intent": intent})
                emit_rag_step('✅', '知识图谱查询完成', f'{entity} · {intent}')
            except Exception as exc:
                emit_rag_step('❌', '知识图谱查询失败', str(exc))
                graph_output = f"知识图谱查询失败：{exc}"
            if isinstance(graph_output, str) and (
                "未返回可用结果" in graph_output
                or "查询失败" in graph_output
                or "未找到关于" in graph_output
            ):
                emit_rag_step('ℹ️', '知识图谱无结果', f'{entity} · {intent} 未找到匹配信息')
                continue
            graph_outputs.append(f"### {entity} · {intent}\n{graph_output}")
            emit_rag_step('📊', '知识图谱返回结果', f'{entity} · {intent} 找到相关信息')

    if graph_outputs:
        sections.append("## 医疗知识图谱\n" + "\n\n".join(graph_outputs))
        emit_rag_step('📚', '知识图谱检索完成', f'共找到 {len(graph_outputs)} 条结果')

    if _should_prefetch_knowledge_base(user_text, graph_outputs):
        try:
            kb_output = search_knowledge_base.invoke(user_text)
            sections.append("## 知识库检索\n" + kb_output)
        except Exception as exc:
            sections.append(f"## 知识库检索\n检索失败：{exc}")

    if not sections:
        return ""

    return (
        "你已经获得以下预检索上下文，这些内容已经是工具检索结果。请只基于这些内容回答，并在必要时说明不确定性。"
        "如果上下文包含“医疗知识图谱”章节，说明知识图谱本轮已经可用且已返回证据，禁止说“知识图谱不可用/暂时不可用”。"
        "如果检索结果与用户问题主题不匹配，请明确说明知识库未检索到相关资料，不要用通用知识补全成确定答案。"
        "不要再次调用任何工具，直接基于上下文作答。\n\n"
        + "\n\n---\n\n".join(sections)
    )

def summarize_old_messages(model, messages: list) -> str:
    """将旧消息总结为摘要"""
    # 提取旧对话
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    # 生成摘要
    summary_prompt = f"""请总结以下对话的关键信息：

{old_conversation}
总结（包含用户信息、重要事实、待办事项）："""

    summary = model.invoke(summary_prompt).content
    return summary


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并返回响应"""
    messages = storage.load(user_id, session_id)

    # 清理可能残留的 RAG 上下文，避免跨请求污染
    get_last_rag_context(clear=True)
    reset_tool_call_guards()
    
    if len(messages) > 50:
        summary = summarize_old_messages(model, messages[:40])

        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    human_message = HumanMessage(content=user_text)
    prefetched_context = _prefetch_medical_context(user_text)

    response_content = ""
    assistant_message = None

    if prefetched_context:
        model_messages = messages + [SystemMessage(content=prefetched_context), human_message]
        assistant_message = model.invoke(model_messages)
        response_content = _message_content_to_text(assistant_message)
    else:
        agent_messages = messages + [human_message]
        result = agent.invoke(
            {"messages": agent_messages},
            config={"recursion_limit": 8},
        )
        if isinstance(result, dict):
            if "output" in result:
                response_content = result["output"]
            elif "messages" in result and result["messages"]:
                assistant_message = result["messages"][-1]
                response_content = _message_content_to_text(assistant_message)
            else:
                response_content = str(result)
        elif hasattr(result, "content"):
            assistant_message = result
            response_content = _message_content_to_text(assistant_message)
        else:
            response_content = str(result)
    
    if assistant_message is None:
        assistant_message = AIMessage(content=response_content)
    messages.extend([human_message, assistant_message])

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    assistant_additional_kwargs = getattr(assistant_message, "additional_kwargs", {}) or {}
    assistant_response_metadata = getattr(assistant_message, "response_metadata", {}) or {}
    extra_message_data = [None] * (len(messages) - 1) + [{
        "rag_trace": rag_trace,
        "additional_kwargs": assistant_additional_kwargs,
        "response_metadata": assistant_response_metadata,
    }]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }


async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并流式返回响应。
    
    架构：使用统一输出队列 + 后台任务，确保 RAG 检索步骤在工具执行期间实时推送，
    而非等待工具完成后才显示。
    """
    messages = storage.load(user_id, session_id)

    # 清理可能残留的 RAG 上下文
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    # 统一输出队列：所有事件（content / rag_step）都汇入这里
    output_queue = asyncio.Queue()

    class _RagStepProxy:
        """代理对象：将 emit_rag_step 的原始 step dict 包装后放入统一输出队列。"""
        def put_nowait(self, step):
            if isinstance(step, dict) and step.get("type"):
                output_queue.put_nowait(step)
            else:
                output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    if len(messages) > 50:
        summary = summarize_old_messages(model, messages[:40])
        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    full_response = ""
    assistant_additional_kwargs: dict = {}
    assistant_response_metadata: dict = {}
    retrieval_done = False

    async def _retrieval_task():
        """执行检索任务，发送检索步骤。"""
        nonlocal retrieval_done
        try:
            await asyncio.to_thread(_prefetch_medical_context, user_text)
        finally:
            retrieval_done = True

    async def _generation_task():
        """执行生成任务，发送内容。"""
        nonlocal messages, full_response, assistant_additional_kwargs, assistant_response_metadata
        
        await retrieval_task
        
        history_messages = list(messages)
        human_message = HumanMessage(content=user_text)
        
        prefetched_context = ""
        rag_context = get_last_rag_context(clear=False)
        if rag_context and "context" in rag_context:
            prefetched_context = rag_context["context"]

        if prefetched_context:
            model_messages = history_messages + [SystemMessage(content=prefetched_context), human_message]
            messages = history_messages + [human_message]
            async for msg in model.astream(model_messages):
                if getattr(msg, "additional_kwargs", None):
                    assistant_additional_kwargs.update(msg.additional_kwargs)
                if getattr(msg, "response_metadata", None):
                    assistant_response_metadata.update(msg.response_metadata)
                content = _message_content_to_text(msg)
                if content:
                    full_response += content
                    await output_queue.put({"type": "content", "data": content})
        else:
            agent_messages = history_messages + [human_message]
            messages = agent_messages
            async for msg, metadata in agent.astream(
                {"messages": agent_messages},
                stream_mode="messages",
                config={"recursion_limit": 8},
            ):
                if not isinstance(msg, AIMessageChunk):
                    continue
                if getattr(msg, "tool_call_chunks", None):
                    continue

                if getattr(msg, "additional_kwargs", None):
                    assistant_additional_kwargs.update(msg.additional_kwargs)
                if getattr(msg, "response_metadata", None):
                    assistant_response_metadata.update(msg.response_metadata)

                content = _message_content_to_text(msg)
                if content:
                    full_response += content
                    await output_queue.put({"type": "content", "data": content})
        await output_queue.put(None)

    # 启动两个任务：检索和生成
    retrieval_task = asyncio.create_task(_retrieval_task())
    generation_task = asyncio.create_task(_generation_task())

    try:
        # 主循环：持续从统一队列取事件并 yield SSE
        while True:
            try:
                event = await asyncio.wait_for(output_queue.get(), timeout=0.1)
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                if retrieval_done and generation_task.done():
                    break
                continue
    except GeneratorExit:
        retrieval_task.cancel()
        generation_task.cancel()
        try:
            await retrieval_task
        except asyncio.CancelledError:
            pass
        try:
            await generation_task
        except asyncio.CancelledError:
            pass
        raise
    finally:
        set_rag_step_queue(None)
        if not retrieval_task.done():
            retrieval_task.cancel()
        if not generation_task.done():
            generation_task.cancel()

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 发送 trace 信息
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # 发送结束信号
    yield "data: [DONE]\n\n"

    # 保存对话
    messages.append(AIMessage(content=full_response))
    extra_message_data = [None] * (len(messages) - 1) + [{
        "rag_trace": rag_trace,
        "additional_kwargs": assistant_additional_kwargs,
        "response_metadata": assistant_response_metadata,
    }]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)
