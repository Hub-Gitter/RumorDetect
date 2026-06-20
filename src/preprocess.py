import re


def preprocess(text: str) -> str:
    """URL → [URL], @user → [USER], #HashTag → 按大小写拆分。"""
    text = str(text)

    # 1. URL → [URL]
    text = re.sub(r'https?://\S+', '[URL]', text)

    # 2. @mention → [USER]
    text = re.sub(r'@\w+', '[USER]', text)

    # 3. Hashtag: 混合大小写→拆分, 全大写/全小写→保留
    def _split_hashtag(m: re.Match) -> str:
        word = m.group(1)
        if word.isupper() or word.islower():
            return word
        # 按大小写边界拆分: "MikeBrown" → "Mike Brown"
        parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+', word)
        return ' '.join(parts) if parts else word

    text = re.sub(r'#(\w+)', _split_hashtag, text)

    return text
