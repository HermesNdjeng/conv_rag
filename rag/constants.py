HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

DEFAULT_MAX_CHUNK_SIZE: int = 2000
DEFAULT_CHUNK_OVERLAP: int = 200

DEFAULT_TOP_K: int = 5
DEFAULT_SCORE_THRESHOLD: float = 0.3

GLOBAL_INDEX: str = "global"
