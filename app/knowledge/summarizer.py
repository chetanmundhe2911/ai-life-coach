from typing import List
from openai import OpenAI
from config import settings


class Summarizer:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def summarize(self, text: str, max_words: int = 200, topic: str = "") -> str:
        if not text.strip():
            return ""
        word_count = len(text.split())
        if word_count <= max_words:
            return text
        topic_hint = f" Focus on aspects related to: {topic}." if topic else ""
        prompt = f"""Summarize the following text in under {max_words} words.
Keep the most important facts and personal details.{topic_hint}
Be concise and factual.

TEXT:
{text}

SUMMARY:"""
        response = self.client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_words * 2,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def summarize_chunks(self, chunks_with_scores: list, topic: str = "") -> str:
        if not chunks_with_scores:
            return ""
        combined = "\n\n".join([chunk.content for chunk, score in chunks_with_scores])
        return self.summarize(combined, topic=topic)