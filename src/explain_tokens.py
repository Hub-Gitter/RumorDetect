"""Gradient × Input (Saliency) 关键词提取。"""

import torch


def get_key_tokens(model, classifier, tokenizer, text: str,
                   target_label: int, top_k: int = 6,
                   device: str = "cpu") -> list:
    """
    用 Gradient × Input 方法计算每个 token 对分类决策的贡献。
    返回 [(token: str, score: float), ...]，按 saliency 分数降序排列。
    text 应为预处理后的文本。
    """
    # 获取 embedding 层
    embedding_layer = model.base_model.embeddings.word_embeddings

    encoded = tokenizer(
        text, padding='max_length', truncation=True,
        max_length=128, return_tensors='pt',
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # 将 input_ids 转为 embedding 并启用梯度
    embeddings = embedding_layer(input_ids)
    embeddings.requires_grad_(True)
    embeddings.retain_grad()

    # PeftModel forward: 需要手动处理 embedding 替代 input_ids
    # 绕过 token embedding 层的 input_ids 输入
    model.eval()
    classifier.eval()

    outputs = model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
    )
    cls_hidden = outputs.last_hidden_state[:, 0, :]
    logits = classifier(cls_hidden)

    logits[0, target_label].backward()
    grad = embeddings.grad  # (1, seq_len, hidden)
    saliency = (grad * embeddings).sum(dim=-1)  # (1, seq_len)
    saliency = saliency.squeeze(0)  # (seq_len,)

    # 转为 token 列表
    token_ids = input_ids[0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    scores = saliency.detach().cpu().tolist()

    # 聚合 subword → word
    token_scores = []
    current_word_tokens = []
    current_word_score = 0.0
    for token, score in zip(tokens, scores):
        if token in {'<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '[PAD]', '[URL]', '[USER]'}:
            continue
        if token.startswith('Ġ'):  # RoBERTa 用 Ġ 表示词首空格
            if current_word_tokens:
                token_scores.append(
                    (''.join(current_word_tokens).replace('Ġ', ''),
                     current_word_score / len(current_word_tokens))
                )
            current_word_tokens = [token]
            current_word_score = score
        else:
            current_word_tokens.append(token)
            current_word_score += score

    # 最后一个词
    if current_word_tokens:
        token_scores.append(
            (''.join(current_word_tokens).replace('Ġ', ''),
             current_word_score / len(current_word_tokens))
        )

    # 按分数降序排列
    token_scores.sort(key=lambda x: x[1], reverse=True)

    return token_scores[:top_k]
