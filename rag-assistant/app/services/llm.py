from openai import APIError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from app.config import settings
from app.models import Chunk, Message


client = OpenAI(api_key=settings.OPENAI_API_KEY)


class LLMServiceError(Exception):
    pass


def generate_answer(question: str, context_chunks: list[Chunk], history: list[Message] | None = None) -> str:
    context_parts: list[str] = []
    for chunk in context_chunks:
        if chunk.page_number is not None:
            context_parts.append(f"[Nguồn: trang {chunk.page_number}]\n{chunk.text}")
        else:
            context_parts.append(chunk.text)

    context = "\n\n---\n\n".join(context_parts)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "Bạn phải trả lời dựa trên context được cung cấp, không bịa thông tin ngoài context.",
        },
    ]

    if history:
        messages.extend(
            {
                "role": message.role,
                "content": message.content,
            }
            for message in history
        )

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nCâu hỏi: {question}",
        }
    )

    try:
        response = client.chat.completions.create(model=settings.OPENAI_MODEL, messages=messages)
    except APITimeoutError as exc:
        raise LLMServiceError("LLM request timed out.") from exc
    except RateLimitError as exc:
        raise LLMServiceError("LLM rate limit exceeded.") from exc
    except APIStatusError as exc:
        raise LLMServiceError("LLM request failed with an API status error.") from exc
    except APIError as exc:
        raise LLMServiceError("LLM request failed due to an API error.") from exc

    return response.choices[0].message.content
