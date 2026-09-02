from hrqde_c.extractors import escoxlm_r


def test_short_text_is_single_chunk():
    assert escoxlm_r._chunks("kurzer Text.") == [(0, "kurzer Text.")]


def test_chunks_cover_text_contiguously_and_respect_limit():
    text = "Satz eins über Pflege. " * 300  # ~6900 Zeichen
    chunks = escoxlm_r._chunks(text)
    assert len(chunks) > 1
    rebuilt = "".join(c for _, c in chunks)
    assert rebuilt == text
    assert all(len(c) <= escoxlm_r.MAX_CHUNK_CHARS for _, c in chunks)
    # Offsets zeigen auf die Originalpositionen
    for offset, chunk in chunks:
        assert text[offset:offset + len(chunk)] == chunk


def test_oversized_piece_is_hard_split():
    text = "x" * 4000  # keine Satzgrenzen
    chunks = escoxlm_r._chunks(text)
    assert "".join(c for _, c in chunks) == text
    assert all(len(c) <= escoxlm_r.MAX_CHUNK_CHARS for _, c in chunks)
