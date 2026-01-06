from pydantic import BaseModel
from datetime import datetime


class ChunkMetadata(BaseModel):
    pdf_name: str
    chunk_number: int | None = None
    page_number: int | None = None
    total_chunks: int | None = None
    all_md_pages: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    gcs_uri: str | None = None


class SearchResult(BaseModel):
    content: str
    metadata: ChunkMetadata
    score: float
    source_index: str
    search_type: str
    vector_field: str | None = None
    model_config = {"frozen": True}


class DocumentContext(BaseModel):
    document_id: str
    context: str
    source_index: str
    pdf_gcs_uri: str | None = None


class DocumentCandidate(BaseModel):
    document_id: str
    chunks: list[SearchResult]
    full_content: str | None = None
    pdf_gcs_uri: str | None = None

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def max_score(self) -> float:
        return max((c.score for c in self.chunks), default=0.0)

    @property
    def avg_score(self) -> float:
        return sum(c.score for c in self.chunks) / len(self.chunks) if self.chunks else 0.0
