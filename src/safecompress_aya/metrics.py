"""Small deterministic metric helpers; full judging requires human rubric."""

def rate(num_positive: int, num_total: int) -> float:
    if num_total <= 0:
        raise ValueError("Need a positive sample count")
    if num_positive < 0 or num_positive > num_total:
        raise ValueError("Positive count outside valid range")
    return num_positive / num_total


def storage_reduction_ratio(reference_bytes: int, compressed_bytes: int) -> float:
    if reference_bytes <= 0 or compressed_bytes <= 0:
        raise ValueError("Both checkpoint sizes must be positive")
    return reference_bytes / compressed_bytes
