import re
import uuid
import secrets
from typing import Dict, Optional

CHAR_LOWER = "abcdefghijklmnopqrstuvwxyz"
NUMBER = "0123456789"
DATA_FOR_RANDOM_STRING = CHAR_LOWER + NUMBER

# 随机字符串


def generate_random_string(length: int) -> str:
    if length < 1:
        raise ValueError("length must be >= 1")

    # secrets 等价 Java SecureRandom
    return "".join(secrets.choice(DATA_FOR_RANDOM_STRING) for _ in range(length))

# 银行卡 Luhn 校验（私有方法对齐）


def _luhn_bank_card_verify(card_number: str) -> bool:
    total = 0
    alternate = False

    for ch in reversed(card_number):
        digit = int(ch)
        if alternate:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
        alternate = not alternate

    return total % 10 == 0

# 文本脱敏


def text_desensitization(
    content: str,
    sensitive_patterns_mapping: Dict[str, str],
) -> str:
    if not content:
        return content

    # 邮箱脱敏（排除 @jd.com）
    email_pattern = re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    for match in email_pattern.finditer(content):
        snippet = match.group()
        if "@jd.com" in snippet:
            continue
        local, _, domain = snippet.partition("@")
        content = content.replace(snippet, f"{local}＠{domain}")

    # 身份证号脱敏
    id_pattern = re.compile(
        r"(?:[^\dA-Za-z_]|^)"
        r"((?:[1-6][1-7]|50|71|81|82)\d{4}"
        r"(?:19|20)\d{2}"
        r"(?:0[1-9]|10|11|12)"
        r"(?:[0-2][1-9]|10|20|30|31)"
        r"\d{3}[0-9Xx])"
        r"(?:[^\dA-Za-z_]|$)"
    )
    for match in id_pattern.finditer(content):
        snippet = match.group(1)
        content = content.replace(snippet, snippet[:12] + "✿✿✿✿✿✿")

    # 手机号脱敏
    phone_pattern = re.compile(
        r"(?:[^\dA-Za-z_]|^)(1[3456789]\d{9})(?:[^\dA-Za-z_]|$)")
    for match in phone_pattern.finditer(content):
        snippet = match.group(1)
        content = content.replace(snippet, snippet[:3] + "✿✿✿✿" + snippet[7:])

    # 银行卡脱敏（+ Luhn 校验）
    bankcard_pattern = re.compile(
        r"(?:[^\dA-Za-z_]|^)(62(?:\d{14}|\d{17}))(?:[^\dA-Za-z_]|$)")
    for match in bankcard_pattern.finditer(content):
        snippet = match.group(1)
        if _luhn_bank_card_verify(snippet):
            content = content.replace(snippet, snippet[:12] + "✿✿✿✿✿✿")

    # 密码 / 自定义敏感词脱敏
    for pattern, word_mapping in sensitive_patterns_mapping.items():
        try:
            start = pattern.index("^)") + 2
            end = pattern.rfind("[^")
        except ValueError:
            content = content.replace(pattern, word_mapping)
            continue

        if start + 1 < end:
            sensitive_word = pattern[start:end]
            compiled = re.compile(pattern)

            for match in compiled.finditer(content):
                snippet = match.group()
                if content.startswith(sensitive_word):
                    content = content.replace(
                        snippet, word_mapping + snippet[-1]
                    )
                else:
                    content = content.replace(
                        snippet, snippet[0] + word_mapping + snippet[-1]
                    )
        else:
            content = content.replace(pattern, word_mapping)

    return content

# 特殊字符移除


def remove_special_chars(input_str: Optional[str]) -> str:
    if not input_str:
        return ""

    special_chars = set(' "&$@=;+?\\{^}%~[]<>#|\'')
    return "".join(c for c in input_str if c not in special_chars)

# UUID


def get_uuid() -> str:
    return str(uuid.uuid4())
