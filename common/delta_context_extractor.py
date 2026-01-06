import re
import logging
from .elastic_retrive_node import DocumentContext, DocumentCandidate

logger = logging.getLogger(__name__)

_PAGE_PATTERN = re.compile(r'page_(\d+)')


def estimate_tokens(text: str) -> int:
    return int(len(text) / 3.3 * 1.1)


def extract_context_delta(
        candidates: list[DocumentCandidate],
        max_tokens: int = 125000,
        initial_padding: int = 25
) -> list[DocumentContext]:
    weights = [c.chunk_count for c in candidates]
    total_weight = sum(weights)

    results = []
    for candidate, weight in zip(candidates, weights):
        budget = int(weight / total_weight * max_tokens)
        context, used_tokens, padding = _extract_with_budget(candidate, budget, initial_padding)
        results.append(context)

        logger.info(
            f"doc={candidate.document_id[:25]} chunks={candidate.chunk_count} "
            f"tokens={used_tokens}/{budget} padding={padding}"
        )

    return results


def _extract_with_budget(
        candidate: DocumentCandidate,
        budget: int,
        initial_padding: int
) -> tuple[DocumentContext, int, int | str]:
    if not candidate.chunks or not candidate.full_content:
        return _empty_context(candidate), 0, 0

    source_index = candidate.chunks[0].source_index
    page_nums = [c.metadata.page_number for c in candidate.chunks if c.metadata.page_number]

    if not page_nums:
        return _full_or_empty(candidate, source_index, budget)

    available = sorted(set(int(m) for m in _PAGE_PATTERN.findall(candidate.full_content)))
    if not available:
        return _full_or_empty(candidate, source_index, budget)

    for padding in [initial_padding, 15, 10, 5, 2, 1, 0]:
        context = _build_context(candidate.full_content, page_nums, available, padding)
        tokens = estimate_tokens(context)

        if tokens > 0 and tokens <= budget:
            return (
                DocumentContext(
                    document_id=candidate.document_id,
                    context=context,
                    source_index=source_index,
                    pdf_gcs_uri=candidate.pdf_gcs_uri
                ),
                tokens,
                padding
            )

    return _extract_raw_chunks(candidate, source_index, budget)


def _extract_raw_chunks(
        candidate: DocumentCandidate,
        source_index: str,
        budget: int
) -> tuple[DocumentContext, int, str]:
    total_chunks = len(candidate.chunks)

    for ratio in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        chunk_limit = max(1, int(total_chunks * ratio))
        chunks_to_try = candidate.chunks[:chunk_limit]

        chunks_text = []
        tokens = 0

        for chunk in chunks_to_try:
            chunk_tokens = estimate_tokens(chunk.content)
            if tokens + chunk_tokens <= budget:
                chunks_text.append(chunk.content)
                tokens += chunk_tokens

        if chunks_text:
            context = "\n\n".join(chunks_text)
            return (
                DocumentContext(
                    document_id=candidate.document_id,
                    context=context,
                    source_index=source_index,
                    pdf_gcs_uri=candidate.pdf_gcs_uri
                ),
                tokens,
                f"-{len(chunks_text)}"
            )

    return _empty_context(candidate), 0, "-0"


def _build_context(content: str, page_nums: list[int], available: list[int], padding: int) -> str:
    ranges = [
        (max(min(available), page - padding), min(max(available), page + padding))
        for page in page_nums
    ]
    merged = _merge_ranges(ranges)

    parts = [
        _extract_pages(content, start, end, max(available))
        for start, end in merged
    ]
    return "\n...\n".join(parts)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    sorted_ranges = sorted(ranges)
    merged = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


def _extract_pages(content: str, start: int, end: int, last_page: int) -> str:
    start_idx = content.find(f"page_{start}")
    if start_idx == -1:
        start_idx = 0

    if end >= last_page:
        return content[start_idx:].strip()

    end_idx = content.find(f"page_{end + 1}", start_idx)
    return content[start_idx:end_idx if end_idx != -1 else len(content)].strip()


def _full_or_empty(
        candidate: DocumentCandidate,
        source_index: str,
        budget: int
) -> tuple[DocumentContext, int, int | str]:
    tokens = estimate_tokens(candidate.full_content)

    if tokens <= budget:
        return (
            DocumentContext(
                document_id=candidate.document_id,
                context=candidate.full_content,
                source_index=source_index,
                pdf_gcs_uri=candidate.pdf_gcs_uri
            ),
            tokens,
            0
        )

    return _extract_raw_chunks(candidate, source_index, budget)


def _empty_context(candidate: DocumentCandidate) -> DocumentContext:
    source_index = candidate.chunks[0].source_index if candidate.chunks else ""
    return DocumentContext(
        document_id=candidate.document_id,
        context="",
        source_index=source_index,
    )


def format_page_tags(content: str) -> str:
    if not content:
        return content

    matches = list(re.finditer(r'page_(\d+)\n?', content))
    if not matches:
        return content

    result = []
    prev_page = None
    last_pos = 0

    for match in matches:
        page_num = match.group(1)

        if prev_page:
            result.append(content[last_pos:match.start()])
            result.append(f"</PAGE {prev_page}>\n")
        else:
            if match.start() > 0:
                result.append(content[last_pos:match.start()])

        result.append(f"<PAGE {page_num}>\n")
        prev_page = page_num
        last_pos = match.end()

    result.append(content[last_pos:])
    if prev_page:
        result.append(f"</PAGE {prev_page}>")

    return ''.join(result)
