from calibre_web2rag.chunking import split_text


def test_split_text_returns_chunks_with_overlap() -> None:
    text = "abcdefghij" * 50
    chunks = split_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert chunks[0][-10:] == chunks[1][:10]


def test_split_text_validates_overlap() -> None:
    try:
        split_text("text", chunk_size=10, overlap=10)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_split_text_respects_sentence_boundaries() -> None:
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = split_text(text, chunk_size=40, overlap=0)
    assert len(chunks) == 2
    assert chunks[0] == "First sentence. Second sentence."
    assert chunks[1] == "Third sentence. Fourth sentence."


def test_split_text_sentence_overlap() -> None:
    text = "Alpha. Bravo. Charlie. Delta. Echo."
    chunks = split_text(text, chunk_size=20, overlap=10)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert not chunk.startswith(" ")


def test_split_text_single_long_sentence_fallback() -> None:
    text = "a" * 200
    chunks = split_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert chunks[0][-10:] == chunks[1][:10]


def test_split_text_empty_input() -> None:
    assert split_text("", chunk_size=100, overlap=10) == []
    assert split_text("   ", chunk_size=100, overlap=10) == []
