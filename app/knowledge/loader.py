"""
app/knowledge/loader.py — Document Loader & Chunker
=====================================================
WHY THIS EXISTS:
  RAG requires your documents to be:
  1. Loaded from disk (txt, pdf, md)
  2. Split into smaller "chunks" (because GPT-4 has a context limit)
  3. Each chunk is later converted to a vector embedding

CONCEPT — WHY CHUNK?
  Imagine your doc is a 50-page book. We can't send the whole book
  to GPT-4 every time. Instead, we split it into ~500 char chunks,
  find the 3 most relevant chunks for the user's question,
  and send only those 3 chunks. Much cheaper & faster!

CONCEPT — CHUNK OVERLAP:
  If chunk 1 ends mid-sentence, chunk 2 starts slightly before
  the end of chunk 1. This "overlap" prevents losing context
  at chunk boundaries.
  
  Example (chunk_size=20, overlap=5):
  Text:    "The quick brown fox jumps over the lazy dog"
  Chunk 1: "The quick brown fox j"
  Chunk 2: "ox jumps over the laz"  ← starts 5 chars before end of chunk 1
  Chunk 3: "lazy dog"
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document

from config import settings
from app.knowledge.utils import get_supported_files, get_file_hash, truncate_text


@dataclass
class LoadedDocument:
    """
    Represents a single chunk after splitting.
    
    We use a dataclass (not a dict) so we get:
    - Type hints
    - Auto-generated __repr__ for easy debugging
    - Immutability via frozen=True if needed
    """
    content: str           # The actual text chunk
    source: str            # Which file it came from
    chunk_index: int       # Which chunk number within the file
    file_hash: str         # MD5 hash to detect file changes
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra info


class DocumentLoader:
    """
    Loads documents from disk and splits them into chunks.
    
    USAGE:
        loader = DocumentLoader()
        chunks = loader.load_all()
        print(f"Loaded {len(chunks)} chunks from {settings.DOCS_DIR}")
    """

    def __init__(self):
        # RecursiveCharacterTextSplitter is LangChain's best general-purpose splitter.
        # It tries to split on paragraphs first, then sentences, then words, then chars.
        # This means chunks end at natural boundaries when possible.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Try these in order
        )

    def load_all(self) -> List[LoadedDocument]:
        """
        Main method: loads ALL documents from the docs directory.
        Returns a flat list of chunks ready for embedding.
        """
        docs_path = settings.docs_path
        files = get_supported_files(docs_path)

        if not files:
            print(f"⚠️  No documents found in {docs_path}")
            print(f"   Add .txt, .md, or .pdf files to get personalized responses")
            return []

        print(f"📂 Found {len(files)} document(s) in {docs_path}")

        all_chunks: List[LoadedDocument] = []
        for file_path in files:
            chunks = self._load_file(file_path)
            all_chunks.extend(chunks)
            print(f"   ✓ {file_path.name} → {len(chunks)} chunks")

        print(f"📄 Total chunks ready for embedding: {len(all_chunks)}")
        return all_chunks

    def _load_file(self, file_path: Path) -> List[LoadedDocument]:
        """
        Loads a single file based on its extension.
        Returns list of chunks from that file.
        """
        file_hash = get_file_hash(str(file_path))
        extension = file_path.suffix.lower()

        try:
            # Load raw LangChain Documents based on file type
            if extension == ".pdf":
                raw_docs = self._load_pdf(file_path)
            else:
                # .txt and .md use the simple TextLoader
                raw_docs = self._load_text(file_path)

            # Split each raw document into chunks
            chunks_docs = self.splitter.split_documents(raw_docs)

            # Convert LangChain Documents → our LoadedDocument dataclass
            loaded = []
            for i, doc in enumerate(chunks_docs):
                loaded.append(LoadedDocument(
                    content=doc.page_content,
                    source=str(file_path),
                    chunk_index=i,
                    file_hash=file_hash,
                    metadata={
                        "filename": file_path.name,
                        "extension": extension,
                        **doc.metadata  # Includes page number for PDFs
                    }
                ))
            return loaded

        except Exception as e:
            print(f"   ❌ Failed to load {file_path.name}: {e}")
            return []

    def _load_text(self, file_path: Path) -> List[Document]:
        """Load .txt or .md files using LangChain's TextLoader"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def _load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load PDF files using PyPDF.
        PyPDFLoader returns one Document per PAGE.
        """
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_text_directly(self, text: str, source: str = "direct_input") -> List[LoadedDocument]:
        """
        WHY: Sometimes users want to add knowledge mid-conversation
        without creating a file. This lets you add text directly.
        
        USAGE:
            loader.load_text_directly("I'm allergic to peanuts", "health_notes")
        """
        doc = Document(page_content=text, metadata={"source": source})
        chunks = self.splitter.split_documents([doc])
        
        return [
            LoadedDocument(
                content=chunk.page_content,
                source=source,
                chunk_index=i,
                file_hash="direct",
                metadata={"source": source}
            )
            for i, chunk in enumerate(chunks)
        ]
