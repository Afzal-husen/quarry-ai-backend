import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document


class DocumentParsingError(Exception):
    """Custom exception raised when a document cannot be parsed successfully."""
    pass


class DocumentParser:
    """Orchestrates parsing of PDF and DOCX files into LangChain Documents."""

    def parse_pdf(self, file_path: Path) -> List[Document]:
        """Parses a PDF file using PyPDFLoader.

        Args:
            file_path: The absolute Path to the PDF file.

        Returns:
            A list of LangChain Document objects.

        Raises:
            DocumentParsingError: If PyPDFLoader fails to load the file.
        """
        try:
            loader = PyPDFLoader(str(file_path))
            return loader.load()
        except Exception as e:
            # Wrap the low-level exception in a custom domain exception
            raise DocumentParsingError(f"Failed to parse PDF file at {file_path}: {str(e)}") from e

    def parse_docx(self, file_path: Path) -> List[Document]:
        """Parses a DOC/DOCX file using Docx2txtLoader.

        Args:
            file_path: The absolute Path to the DOC/DOCX file.

        Returns:
            A list of LangChain Document objects.

        Raises:
            DocumentParsingError: If Docx2txtLoader fails to load the file.
        """
        try:
            loader = Docx2txtLoader(str(file_path))
            return loader.load()
        except Exception as e:
            # Wrap the low-level exception in a custom domain exception
            raise DocumentParsingError(f"Failed to parse Word file at {file_path}: {str(e)}") from e

    def parse_file(self, file_path: Path) -> List[Document]:
        """Parses a file based on its extension.

        Args:
            file_path: The absolute Path to the file.

        Returns:
            A list of LangChain Document objects.

        Raises:
            DocumentParsingError: If the extension is unsupported or parsing fails.
        """
        if not file_path.exists():
            raise DocumentParsingError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self.parse_pdf(file_path)
        elif suffix in [".docx", ".doc"]:
            return self.parse_docx(file_path)
        else:
            raise DocumentParsingError(f"Unsupported file format '{suffix}' for file: {file_path.name}")
