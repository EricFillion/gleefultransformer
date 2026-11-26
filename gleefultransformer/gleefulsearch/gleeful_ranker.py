from dataclasses import dataclass, field
import re

from gleefultransformer.gleefulsearch.utils import (
    GleefulDocument,
    RankerResult,
    es_get_device,
    es_get_logger,
    split_sentences,
)


@dataclass
class RankerCallArgs:
    bs: int = 32
    limit: int = 1


# Internal dataclasses.
@dataclass
class Sentence:
    text: str
    doc_idx: int
    paragraph_idx: int
    score: float = 0


@dataclass
class Paragraph:
    text: str
    doc_idx: int
    paragraph_idx: int
    doc_score: float
    paragraph_crossencoder_score: float = 0
    sentences: list[Sentence] = field(default_factory=list)
    best_sentence: Sentence | None = None
    aggregated_score: float = 0


class GleefulRanker:
    ignore_original_score: bool

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ModuleNotFoundError as error:
            if error.name != "sentence_transformers":
                raise
            raise ImportError(
                "GleefulRanker requires Sentence Transformers. "
                "Install it with `pip install 'gleefultransformer[search]'`."
            ) from error

        self.ignore_original_score = False

        self.device = es_get_device()

        self.model = CrossEncoder(model_name, device=str(self.device))

        self.logger = es_get_logger()

    @staticmethod
    def get_blocks(
        gleeful_searchset_result: list[GleefulDocument],
    ) -> tuple[list[Paragraph], list[Sentence]]:
        # Each input doc gets one block. So output blocks align to input docs.

        kept_paragraphs = []
        kept_sentences = []

        for doc_idx, doc_search_result in enumerate(gleeful_searchset_result):
            text = doc_search_result.text
            doc_score = doc_search_result.score

            text = text.replace("\r\n", "\n").replace("\r", "\n")
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            for paragraph_idx, paragraph in enumerate(paragraphs):
                if len(paragraph.split()) > 5:
                    kept_paragraphs.append(
                        Paragraph(
                            text=paragraph,
                            doc_idx=doc_idx,
                            paragraph_idx=paragraph_idx,
                            doc_score=doc_score,
                        )
                    )

                    sentences = split_sentences(paragraph)
                    for sentence in sentences:
                        if len(sentence.split()) > 5:
                            kept_sentences.append(
                                Sentence(
                                    text=sentence,
                                    doc_idx=doc_idx,
                                    paragraph_idx=paragraph_idx,
                                )
                            )
                            kept_paragraphs[-1].sentences.append(kept_sentences[-1])
        return kept_paragraphs, kept_sentences

    def __call__(
        self,
        text: str,
        docs: list[GleefulDocument],
        args: RankerCallArgs | None = None,
    ) -> list[RankerResult]:
        if not docs:
            return []

        if args is None:
            args = RankerCallArgs()
        paragraphs, sentences = self.get_blocks(docs)
        if not paragraphs:
            return []

        paragraph_texts = dict.fromkeys(paragraph.text for paragraph in paragraphs)
        sentence_texts = dict.fromkeys(
            sentence.text for sentence in sentences if sentence.text not in paragraph_texts
        )
        self.logger.debug(
            "GleefulRanker scoring %d unique pairs", len(paragraph_texts) + len(sentence_texts)
        )

        scores = {}
        # Keep the two lengths in separate batches to limit padding.
        for passages in (paragraph_texts, sentence_texts):
            if passages:
                predicted_scores = self.model.predict(
                    [(text, passage) for passage in passages],
                    batch_size=args.bs,
                    show_progress_bar=False,
                )
                scores.update(zip(passages, predicted_scores))
        for paragraph in paragraphs:
            paragraph.paragraph_crossencoder_score = scores[paragraph.text]
        for sentence in sentences:
            sentence.score = scores[sentence.text]

        for paragraph in paragraphs:
            paragraph.best_sentence = max(paragraph.sentences, key=lambda s: s.score, default=None)
            paragraph.aggregated_score = (
                (0.6 * paragraph.doc_score) * (not self.ignore_original_score)
                + 0.3 * paragraph.paragraph_crossencoder_score
            )
            if paragraph.best_sentence is not None:
                paragraph.aggregated_score += 0.1 * paragraph.best_sentence.score

        # Rank the paragraphs and return results.
        paragraphs = sorted(paragraphs, key=lambda p: p.aggregated_score, reverse=True)
        return [
            RankerResult(
                text=paragraph.text,
                score=float(paragraph.aggregated_score),
                best_sentence=(
                    paragraph.best_sentence.text if paragraph.best_sentence else ""
                ),
                metadata=docs[paragraph.doc_idx].metadata,
            )
            for paragraph in paragraphs[: args.limit]
        ]
