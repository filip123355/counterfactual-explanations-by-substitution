def assert_not_none[T](val: T | None) -> T:
    if val is None:
        raise ValueError("Expected value to be not None")
    return val
