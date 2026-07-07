from types import SimpleNamespace
from unittest.mock import patch

from app.models import Chunk
from app.services.llm import generate_answer


def test_generate_answer_builds_context_and_returns_content():
    chunks = [
        Chunk(text="Đây là chunk có số trang", page_number=3),
        Chunk(text="Đây là chunk không có số trang", page_number=None),
        Chunk(text="Chunk phụ trợ", page_number=7),
    ]

    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="câu trả lời giả lập",
                )
            )
        ]
    )

    with patch("app.services.llm.client.chat.completions.create", return_value=mock_response) as mock_create:
        result = generate_answer(question="Câu hỏi test", context_chunks=chunks)

    assert result == "câu trả lời giả lập"
    mock_create.assert_called_once()

    _, kwargs = mock_create.call_args
    messages = kwargs["messages"]

    assert kwargs["model"]
    assert "Câu hỏi test" in messages[1]["content"]
    assert "Đây là chunk có số trang" in messages[1]["content"]
    assert "Đây là chunk không có số trang" in messages[1]["content"]
    assert "Chunk phụ trợ" in messages[1]["content"]
    assert "[Nguồn: trang" in messages[1]["content"]
