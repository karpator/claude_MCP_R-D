import os
from typing import AsyncIterable

import aiostream.stream
from jinja2 import Environment, FileSystemLoader


language_dict = {
    "en": "Sources",
    "hu": "Források",
}


class AsyncSourceFormatting():
    async def prep_async(self, shared):
        return {
            "documents": shared.context.documents,
        }

    async def exec_async(self, prep_res):
        documents = prep_res["documents"]

        parsed_documents_set = {doc.pdf_gcs_uri: doc for doc in documents}
        parsed_documents = [{
            "gcs_uri": doc.pdf_gcs_uri or "gs://these/arent/the/pdfs/youre/looking/for.pdf",
            # TODO: Attila will upload the data to ElasticSearch
            "pdf_name": doc.document_id,
        } for doc in parsed_documents_set.values()]

        return parsed_documents

    async def post_async(self, shared, prep_res, exec_res):
        shared.output.documents = exec_res
        return "default"


@deprecated
class AsyncSourceFormattingOld():
    async def prep_async(self, shared):
        streaming = isinstance(shared.output.answer, AsyncIterable)
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = Environment(loader=FileSystemLoader(current_dir), enable_async=True)

        parsed_documents_set = {doc["pdf_gcs_uri"]: doc for doc in shared.context.documents}
        parsed_documents = [{
            "gcs_uri": doc["pdf_gcs_uri"],
            "pdf_name": doc["pdf_name"],
        } for doc in parsed_documents_set.values()]
        parsed_documents_str = await env.get_template("sources.jinja2").render_async(
            documents=parsed_documents
        )

        language = shared.context.language if shared.context.language in language_dict else "en"

        return {
            "parsed_documents": parsed_documents_str,
            "streaming": streaming,
            "answer": shared.output.answer,
            "language": language,
        }

    async def exec_async(self, prep_res):
        answer = prep_res["answer"]
        if prep_res["streaming"]:
            async def stream():
                yield {"text": f"\n\n{language_dict[prep_res['language']]}:\n\n{prep_res['parsed_documents']}"}

            merged = aiostream.stream.chain(answer, stream())
            return merged
        else:
            answer["text"] += f"\n\n{language_dict[prep_res['language']]}:\n\n{prep_res['parsed_documents']}"
            return answer

    async def post_async(self, shared, prep_res, exec_res):
        shared.output.answer = exec_res
        return "default"
