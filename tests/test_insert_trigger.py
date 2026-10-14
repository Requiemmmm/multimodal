from morph.suffix.insert import insert_trigger


def test_user_suffix_default():
    assert insert_trigger("prompt", " δ") == "prompt δ"


def test_positions():
    assert insert_trigger("abcd", "X", "prefix") == "Xabcd"
    assert insert_trigger("abcd", "X", "infix", infix_at=2) == "abXcd"


def test_invalid_infix():
    try:
        insert_trigger("abcd", "X", "infix", infix_at=9)
    except ValueError as exc:
        assert "infix_at" in str(exc)
    else:
        raise AssertionError("expected boundary error")
