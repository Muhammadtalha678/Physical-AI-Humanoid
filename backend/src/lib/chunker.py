import re


def chunk_markdown(text: str, chunk_size=2000):
    text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks