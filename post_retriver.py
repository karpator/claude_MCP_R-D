from typing import Any

from application_logging import ApplicationLogging

from common import extract_context_delta
from common.delta_context_extractor import format_page_tags
from common.elastic_retrive_node import DocumentContext
from helpers import DatabaseConfiguration
from pocketflows import AsyncNode


logger = ApplicationLogging.depends()


class AsyncPostRetriever(AsyncNode):

    async def exec_async(self, documents) -> dict[str, list[DocumentContext]]:


        if not documents:
            return {"documents": []}

        try:
            extracted = extract_context_delta(documents)

            return {"documents": [
                DocumentContext(
                    document_id=doc.document_id,
                    context=format_page_tags(doc.context),
                    source_index=doc.source_index,
                    pdf_gcs_uri=DatabaseConfiguration.create_gcs_path(
                        db_type=DatabaseConfiguration.determine_database_from_index(doc.source_index),
                        file_name=doc.document_id
                    )
                )
                for doc in extracted
            ]}

        except Exception as e:
            logger.error("Context extraction failed", extra={"error": str(e), "doc_count": len(documents)})
            return {"documents": []}

