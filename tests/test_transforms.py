from harm_route.transforms import srp_cyrl_to_latn


def test_srp_cyrl_to_latn():
    assert srp_cyrl_to_latn("Љубав и њежност") == "Ljubav i nježnost"
    assert srp_cyrl_to_latn("Ђорђе, Џон, ћуп, жута чаша.") == "Đorđe, Džon, ćup, žuta čaša."
    assert srp_cyrl_to_latn("abc 123 ?") == "abc 123 ?"  # non-Cyrillic untouched


def test_srp_alphabet_complete():
    # all 30 letters of Serbian Cyrillic, lower and upper, are mapped (none left in the output)
    letters = "абвгдђежзијклљмнњопрстћуфхцчџш"
    out = srp_cyrl_to_latn(letters + letters.upper())
    assert not any("Ѐ" <= ch <= "ӿ" for ch in out)
