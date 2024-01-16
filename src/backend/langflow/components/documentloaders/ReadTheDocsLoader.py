from langflow import CustomComponent
from typing import Dict, Optional,List
from langchain_core.documents import Document
from langchain_community.document_loaders.readthedocs import ReadTheDocsLoader


class ReadTheDocsLoaderComponent(CustomComponent):
    display_name = "ReadTheDocsLoader"
    description = "Load `ReadTheDocs` documentation directory."

    def build_config(self):
        return {
            "metadata": {"display_name": "Metadata", "default": {},"field_type": "dict"},
            "path": {"display_name": "Local directory", "required": True},
        }

    def build(
        self,
        path: str,
        metadata: Optional[Dict] = None,
    ) -> List[Document]:
        documents = ReadTheDocsLoader(path=path).load()
        if(metadata):
            for document in documents:
                if not document.metadata:
                    document.metadata = metadata
                else:
                    document.metadata.update(metadata)
        return documents