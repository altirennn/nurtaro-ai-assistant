from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed(self, text: str) -> list[float]: ...


_CONCEPTS = {
    **{word: "toy_block" for word in ("кубик", "кубики", "кубиков", "кубиками", "кубике", "текше", "текшелер", "текшені", "текшемен")},
    **{word: "reading" for word in ("книга", "книги", "книгу", "книжка", "книжку", "книжкой", "кітап", "кітапты", "кітаппен")},
    **{word: "play" for word in ("игра", "игру", "игрой", "играть", "игрушка", "ойын", "ойынды", "ойынмен", "ойнау", "ойнағанда")},
    **{word: "sort" for word in ("сортировать", "сортировка", "сұрыптау", "сұрыптауға")},
    **{word: "color" for word in ("цвет", "цвета", "цвета", "форму", "түстер", "түсін", "пішінін")},
    **{word: "water" for word in ("вода", "водой", "воды", "су", "сумен", "суды")},
    **{word: "fabric" for word in ("ткань", "тканью", "тканью", "мата", "матамен", "матаны")},
    **{word: "fabric" for word in ("матадағы", "матаның", "матамен")},
    **{word: "sensation" for word in ("ощущение", "ощущения", "сезім", "сезімді", "сезімдерді", "сезіммен")},
    **{word: "ball" for word in ("мяч", "мяча", "доп", "доппен")},
    **{word: "child" for word in ("ребенок", "ребёнок", "ребёнка", "ребенком", "малыш", "малыша", "бала", "балаға", "баламен", "балам")},
    **{word: "adult" for word in ("взрослый", "взрослым", "взрослого", "взрослым", "ересек", "ересек", "ересекпен")},
    **{word: "parent" for word in ("родитель", "родители", "ата-ана", "ата-аналар")},
    **{word: "safety" for word in ("безопасность", "безопасно", "қауіпсіздік", "қауіпсіз")},
    **{word: "reading" for word in ("читать", "чтение", "читай", "оқу", "оқуды", "оқимыз", "оқуды")},
}

_STOP_WORDS = set("а без бы в во для до же за и из или к ко но о об от по при с со у что чтобы как это эта этот эти на не ни мы вы он она они его ее её их можно ли если когда тогда там здесь вместе какой какая какие какой-то будет должен может могут под над между очень уже ещё еще потом или же предложить предложите предложить ребенку ребёнку ребенка ребёнка ребёнку ребёнком ребенком малыш малышу малыша малышу".split())
_STOP_WORDS.update("ал бірақ және мен сен сіз ол бұл сол үшін туралы қалай қандай не болады бала балаға баланы баламен бірге кезде кейін бір осы ұсыну ұсыныңыз ұсынуға".split())


def lexical_terms(text: str) -> set[str]:
    tokens = re.findall(r"[a-zа-яёәғқңөұүһі0-9]+", text.lower())
    terms: set[str] = set()
    for token in tokens:
        if token in _STOP_WORDS or len(token) < 3:
            continue
        concept = _CONCEPTS.get(token)
        if concept:
            terms.add(concept)
            continue
        # A small suffix stemmer supplies common RU/KK inflections without a dependency.
        stem = token
        for suffix in sorted(("ларымыз", "леріміз", "лары", "лері", "лар", "лер", "дың", "дің", "тың", "тің", "мен", "бен", "пен", "ға", "ге", "ты", "ті", "ды", "ді", "дан", "ден", "тан", "тен", "ами", "ями", "ого", "его", "ыми", "ими", "ать", "ять", "ить", "еть", "ую", "юю", "ами", "ями", "ов", "ев", "ей", "ой", "ом", "ем", "ам", "ям", "ах", "ях", "а", "я", "ы", "и", "у", "ю", "е", "о"), key=len, reverse=True):
            if stem.endswith(suffix) and len(stem) - len(suffix) >= 4:
                stem = stem[:-len(suffix)]
                break
        terms.add(_CONCEPTS.get(stem, stem))
    return terms


class HashingEmbedder:
    """Deterministic local word/subword vectors with a small RU-KK concept bridge."""

    name = "hashing-ngram"

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-zа-яёәғқңөұүһі0-9]+", text.lower())
        features: dict[int, float] = defaultdict(float)
        expanded = tokens + [term for term in lexical_terms(text) if term not in tokens]
        for token in expanded:
            idx = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16) % self.dimensions
            features[idx] += 1.0
            if len(token) > 3:
                padded = f"^{token}$"
                for width in (3, 4):
                    for i in range(max(0, len(padded) - width + 1)):
                        gram = padded[i : i + width]
                        gram_idx = int(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest(), 16) % self.dimensions
                        features[gram_idx] += 0.04
        vector = [0.0] * self.dimensions
        norm = math.sqrt(sum(value * value for value in features.values())) or 1.0
        for index, value in features.items():
            vector[index] = value / norm
        return vector


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


def create_embedder(backend: str, model_name: str) -> Embedder:
    if backend in {"sentence-transformers", "auto"}:
        try:
            return SentenceTransformerEmbedder(model_name)
        except (ImportError, OSError, RuntimeError) as exc:
            if backend == "sentence-transformers":
                raise RuntimeError(
                    "Для EMBEDDING_BACKEND=sentence-transformers установите requirements-semantic.txt "
                    "и скачайте модель multilingual-e5-small"
                ) from exc
    return HashingEmbedder()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0
