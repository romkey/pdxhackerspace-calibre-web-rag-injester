from calibre_web2rag.chunking import split_text


def test_split_text_returns_chunks_with_overlap() -> None:
    text = "abcdefghij" * 50
    chunks = split_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert chunks[0][-10:] == chunks[1][:10]


def test_split_text_validates_overlap() -> None:
    try:
        split_text("text", chunk_size=10, overlap=10)
        assert False, "Expected ValueError"
    except ValueError:
        assert True
