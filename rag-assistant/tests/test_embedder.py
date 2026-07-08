from unittest.mock import patch

import numpy as np

from app.services.embedder import embed_texts


def test_embed_texts_uses_cache_for_repeated_text():
    def fake_encode(texts, normalize_embeddings=True):
        assert normalize_embeddings is True
        return np.array(
            [
                [0.5, 0.5],
                [0.7, 0.7],
            ]
        )

    with patch("app.services.embedder._model.encode", side_effect=fake_encode) as mock_encode:
        result = embed_texts(["xin chào", "tạm biệt", "xin chào"])

    mock_encode.assert_called_once()
    args, kwargs = mock_encode.call_args
    assert args[0] == ["xin chào", "tạm biệt"]
    assert kwargs["normalize_embeddings"] is True
    assert len(result) == 3
    assert result[0] == result[2]


def test_embed_texts_calls_model_again_for_new_text_after_cache_hit():
    def fake_encode(texts, normalize_embeddings=True):
        assert normalize_embeddings is True
        return np.array(
            [[float(index) + 0.5, float(index) + 0.5] for index, _ in enumerate(texts)]
        )

    with patch("app.services.embedder._model.encode", side_effect=fake_encode) as mock_encode:
        embed_texts(["câu một"])
        embed_texts(["câu một", "câu hai"])

    assert mock_encode.call_count == 2
    first_call_args, first_call_kwargs = mock_encode.call_args_list[0]
    second_call_args, second_call_kwargs = mock_encode.call_args_list[1]

    assert first_call_args[0] == ["câu một"]
    assert first_call_kwargs["normalize_embeddings"] is True
    assert second_call_args[0] == ["câu hai"]
    assert second_call_kwargs["normalize_embeddings"] is True
