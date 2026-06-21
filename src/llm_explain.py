"""LLM 解释生成：调用 SJTU DeepSeek V3.2 API。"""

import os
import time
import requests

# 加载 .env 文件
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()

API_URL = "https://models.sjtu.edu.cn/api/v1/chat/completions"
API_KEY = os.environ.get("SJTU_API_KEY", "")

SYSTEM_PROMPT = (
    "You are a social media rumor detection expert. Your task is to explain "
    "why a tweet was classified as rumor or non-rumor by analyzing its "
    "specific linguistic choices.\n\n"
    "CRITICAL REQUIREMENTS:\n"
    "1. You MUST quote at least TWO specific words or phrases directly from "
    "the tweet as concrete evidence, using quotation marks. Explanations "
    "without quoted words from the tweet are INVALID.\n"
    "2. Write 2-3 sentences ONLY. Start directly with analysis — do NOT "
    "prepend \"Rumor:\", \"Non-rumor:\", \"RUMOR:\", \"NON-RUMOR:\", "
    "or any label prefix. No greetings, no bullet points.\n"
    "3. Focus on WHY the specific wording signals (non-)rumor — do NOT just "
    "repeat the predicted label or describe what the tweet is about.\n\n"
    "For RUMOR: point out speculative language (\"reportedly\", \"BREAKING\", "
    "\"unconfirmed\"), emotional exaggeration (\"horrific\", \"worst ever\"), "
    "anonymous sources (\"people are saying\"), or claims presented as fact "
    "without attribution.\n\n"
    "For NON-RUMOR: point out official sources (\"police confirmed\", "
    "\"according to\"), specific named entities (organizations, locations), "
    "measured language, or factual reporting tone without inflammatory words.\n\n"
    "When classifier confidence is low (<70%): acknowledge that the tweet "
    "contains mixed signals, and explain which elements point each way.\n\n"
    "When classifier confidence is high (>90%): state clearly which specific "
    "language patterns make this a clear-cut case.\n\n"
    "DO NOT:\n"
    "- Restate the label or confidence at the start (\"Rumor:\", \"Non-rumor:\")\n"
    "- Use markdown formatting (**bold**, *italic*) in your explanation\n"
    "- Mention importance scores or classifier internals (\"the model thinks…\")\n"
    "- Use generic templates that could apply to any tweet\n"
    "- Invent sources or facts not present in the tweet\n\n"
    "EXAMPLES (note: no label prefix, direct analysis):\n"
    '- The all-caps "BREAKING" creates false urgency '
    'without verification, and "officer shot the teen" states an unverified '
    'allegation as fact with no source cited. The absence of any official '
    'attribution is a strong rumor indicator.\n'
    '- "Police confirmed" directly cites an '
    'official law enforcement source, and "according to the report" anchors '
    'the claim in documentation. The neutral, factual phrasing avoids '
    'emotional manipulation.\n'
    '- While "police statement" suggests an official '
    'source, the phrase "it is rumored that" introduces unverified information. '
    'The tweet mixes credible attribution with speculative content.'
)


def _call_api(messages: list, model: str = "deepseek-chat",
              max_tokens: int = 150, temperature: float = 0.3,
              max_retries: int = 5) -> str:
    """调用 SJTU API，带重试逻辑。"""
    if not _is_valid_key(API_KEY):
        raise RuntimeError("Invalid or placeholder API key")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL, json=payload, headers=headers, timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                time.sleep(10)
            elif resp.status_code >= 500:
                time.sleep(3)
            else:
                raise RuntimeError(
                    f"API error {resp.status_code}: {resp.text}"
                )
        except requests.exceptions.Timeout:
            time.sleep(3)
        except requests.exceptions.ConnectionError:
            time.sleep(5)
        except requests.exceptions.JSONDecodeError:
            time.sleep(3)

    raise RuntimeError(f"API call failed after {max_retries} retries")


def fallback_explain(label: int, key_tokens: list) -> str:
    """API 不可用时的模板解释。"""
    label_name = "Rumor" if label == 1 else "Non-rumor"
    tokens_str = ", ".join([t[0] if isinstance(t, tuple) else str(t)
                            for t in key_tokens[:4]])
    return (
        f"Classified as {label_name}. "
        f"Key evidence: '{tokens_str}'. "
        f"(Note: This is a template-based explanation. "
        f"The LLM-powered explanation is unavailable due to API issues.)"
    )


def _build_user_message(text: str, label: int, confidence: float,
                        key_tokens: list) -> str:
    label_name = "Rumor" if label == 1 else "Non-rumor"
    token_names = []
    for t in key_tokens[:6]:
        name = t[0] if isinstance(t, tuple) else str(t)
        token_names.append(name)
    tokens_str = '", "'.join(token_names)
    confidence_note = ""
    if confidence < 0.7:
        confidence_note = (
            "\nNote: confidence is relatively low ({:.0f}%) — the tweet may "
            "contain mixed signals. If so, acknowledge both sides in your "
            "explanation."
        ).format(confidence * 100)
    elif confidence > 0.9:
        confidence_note = (
            "\nNote: confidence is high ({:.0f}%) — the language patterns "
            "are clear-cut. Explain what makes this unambiguous."
        ).format(confidence * 100)
    return (
        f'Tweet: "{text}"\n\n'
        f"Classifier prediction: {label_name} ({confidence * 100:.0f}% confidence)\n"
        f'Key evidence words: "{tokens_str}"'
        f"{confidence_note}\n\n"
        f"Explain the classification in 2-3 sentences, quoting specific words "
        f"from the tweet as evidence:"
    )


def _is_valid_key(key: str) -> bool:
    """检查 API key 是否看起来像真实 key（非空、非模板、全 ASCII）。"""
    if not key:
        return False
    try:
        key.encode('latin-1')
    except UnicodeEncodeError:
        return False
    # 排除模板占位符
    if '填写' in key or 'YOUR_API_KEY' in key.upper():
        return False
    return True


def generate_explanation(text: str, prediction_label: int,
                         confidence: float, key_tokens: list) -> str:
    """生成 2-3 句自然语言解释。API 不可用时自动降级。"""
    if not _is_valid_key(API_KEY):
        return fallback_explain(prediction_label, key_tokens)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(
            text, prediction_label, confidence, key_tokens
        )},
    ]

    try:
        return _call_api(messages)
    except (RuntimeError, requests.exceptions.RequestException):
        return fallback_explain(prediction_label, key_tokens)


def batch_generate_explanations(items: list, batch_size: int = 5) -> list:
    """批量生成解释。items 是 (text, label, confidence, key_tokens) 元组列表。"""
    explanations = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        if len(batch) == 1:
            # 单条不用批处理 prompt
            text, label, conf, tokens = batch[0]
            exp = generate_explanation(text, label, conf, tokens)
            explanations.append(exp)
        else:
            # 合并成批量 prompt
            batch_parts = []
            for j, (text, label, conf, tokens) in enumerate(batch):
                label_name = "Rumor" if label == 1 else "Non-rumor"
                token_names = [t[0] if isinstance(t, tuple) else str(t)
                               for t in tokens]
                tokens_str = ", ".join(f'"{t}"' for t in token_names)
                conf_str = f"{conf*100:.0f}%"
                if conf < 0.7:
                    conf_str += " (low — check for mixed signals)"
                elif conf > 0.9:
                    conf_str += " (high — clear-cut case)"
                batch_parts.append(
                    f"Tweet {j+1}: \"{text}\"\n"
                    f"  Prediction: {label_name} ({conf_str})\n"
                    f"  Key evidence: {tokens_str}"
                )

            batch_prompt = (
                "For each tweet below, explain its classification in 2-3 "
                "sentences. QUOTE at least two specific words/phrases from "
                "each tweet as evidence. For RUMOR: identify speculative "
                "language, lack of sources, or emotional exaggeration. For "
                "NON-RUMOR: identify official sources, specific entities, "
                "or factual tone. When confidence is low, note mixed signals. "
                "Number your responses 1, 2, 3, ... with NO greetings or "
                "prefixes:\n\n"
                + "\n\n".join(batch_parts)
            )

            try:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": batch_prompt},
                ]
                result = _call_api(messages, max_tokens=150 * len(batch))
                # 按编号拆分，支持 "1. " / "1) " / "1: " 三种格式
                import re
                batch_exps = []
                for line in result.split('\n'):
                    m = re.match(r'^(\d+)[\.\)\:]\s+', line.strip())
                    if m:
                        idx = int(m.group(1)) - 1  # convert to 0-based
                        text = re.sub(r'^\d+[\.\)\:]\s+', '', line.strip())
                        while len(batch_exps) <= idx:
                            batch_exps.append('')
                        batch_exps[idx] = text
                # 如果拆分失败，用正文双换行作为回退分隔
                if len(batch_exps) < len(batch):
                    blocks = [b.strip() for b in result.split('\n\n') if b.strip()]
                    if len(blocks) >= len(batch):
                        batch_exps = blocks
                # 仍然不足则对缺位逐条 fallback
                while len(batch_exps) < len(batch):
                    t, l, c, tok = batch[len(batch_exps)]
                    batch_exps.append(fallback_explain(l, tok))
                explanations.extend(batch_exps[:len(batch)])
            except (RuntimeError, requests.exceptions.RequestException):
                for text, label, conf, tokens in batch:
                    explanations.append(fallback_explain(label, tokens))

        # 速率控制：10 RPM
        time.sleep(6)

        if (i + batch_size) % 50 == 0:
            print(f"  LLM progress: {min(i + batch_size, len(items))}/{len(items)}")

    return explanations
