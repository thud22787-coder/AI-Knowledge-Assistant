from openai import OpenAI

from app.config import settings
from app.models import Chunk


client = OpenAI(api_key=settings.OPENAI_API_KEY)


def generate_answer(question: str, context_chunks: list[Chunk]) -> str:
    context_parts: list[str] = []
    for chunk in context_chunks:
        if chunk.page_number is not None:
            context_parts.append(f"[Nguồn: trang {chunk.page_number}]\n{chunk.text}")
        else:
            context_parts.append(chunk.text)

    context = "\n\n---\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Bạn phải trả lời dựa trên context được cung cấp, không bịa thông tin ngoài context.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nCâu hỏi: {question}",
            },
        ],
    )

    return response.choices[0].message.content
