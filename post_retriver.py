import re
from typing import Any

from application_logging import ApplicationLogging

from common import extract_context_delta
from common.delta_context_extractor import format_page_tags
from common.elastic_retrive_node import DocumentContext
from pocketflows import AsyncNode


logger = ApplicationLogging.depends()


class AsyncPostRetriever(AsyncNode):


    async def exec_async(self, documents) -> dict[str, list[DocumentContext]]:


        if not documents:
            return {"documents": []}

        try:
            extracted = extract_context_delta(documents)

            rsp_docs = []
            for doc in extracted:
                gcs_uri = doc.pdf_gcs_uri
                # Pattern to match _page_X where X is one or more digits at the end before extension
                pattern = r'_page_\d+(\.[^.]+)$'
                
                if not gcs_uri:
                    logger().warning(f"Document missing pdf_gcs_uri for document_id: {doc.document_id}")
                    continue
            
                # Replace the pattern with just the extension
                fixed_uri = re.sub(pattern, r'\1', gcs_uri)
                
                
                rsp_doc = DocumentContext(
                    document_id=doc.document_id,
                    context=format_page_tags(doc.context),
                    source_index=doc.source_index,
                    pdf_gcs_uri=fixed_uri,
                )
                rsp_docs.append(rsp_doc)

            return {"documents": rsp_docs}

        except Exception as e:
            logger().exception("Context extraction failed", extra={"error": str(e), "doc_count": len(documents)})
            return {"documents": []}
