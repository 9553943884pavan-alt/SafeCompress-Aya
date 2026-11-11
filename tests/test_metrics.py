from safecompress_aya.metrics import rate, storage_reduction_ratio


def test_rate():
    assert rate(2, 5) == 0.4


def test_compression_ratio():
    assert storage_reduction_ratio(100, 25) == 4
