"""SGML document parser and XML topic parser."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    docno: str
    full_text: str
    title_text: str  # text from TITLE/HD/HEADING/LD/KH fields


@dataclass
class Topic:
    qid: str
    title: str
    desc: str
    narr: str


# Tags containing headline/title content per language
_TITLE_TAGS_EN = {"HD", "LD", "KH"}
_TITLE_TAGS_CS = {"TITLE", "HEADING"}

_DOC_RE = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
_DOCNO_RE = re.compile(r"<DOCNO>(.*?)</DOCNO>", re.DOTALL)
_TAG_RE = re.compile(r"<(/?)(\w+)[^>]*>")
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")


def _extract_field_text(block: str, tag_names: set[str]) -> str:
    """Extract text content from specific SGML tags."""
    parts = []
    for tag_name in tag_names:
        pattern = re.compile(
            rf"<{tag_name}>(.*?)</{tag_name}>", re.DOTALL | re.IGNORECASE
        )
        for m in pattern.finditer(block):
            cleaned = _STRIP_TAGS_RE.sub(" ", m.group(1))
            parts.append(cleaned)
    return " ".join(parts)


def _strip_all_tags(text: str) -> str:
    """Remove all SGML/XML tags from text."""
    return _STRIP_TAGS_RE.sub(" ", text)


def parse_documents(
    file_list_path: str, data_dir: str, lang: str, extract_fields: bool = False
):
    """Parse documents from SGML files listed in file_list_path.

    Yields Document objects.
    """
    encoding = "iso-8859-1" if lang == "en" else "utf-8"
    title_tags = _TITLE_TAGS_EN if lang == "en" else _TITLE_TAGS_CS

    base = Path(data_dir)
    doc_dir = base / f"documents_{lang}"

    with open(base / file_list_path, encoding="utf-8") as f:
        file_names = [line.strip() for line in f if line.strip()]

    for fname in file_names:
        fpath = doc_dir / fname
        with open(fpath, encoding=encoding, errors="replace") as f:
            content = f.read()

        for doc_match in _DOC_RE.finditer(content):
            block = doc_match.group(1)
            docno_match = _DOCNO_RE.search(block)
            if not docno_match:
                continue
            docno = docno_match.group(1).strip()
            full_text = _strip_all_tags(block)

            title_text = ""
            if extract_fields:
                title_text = _extract_field_text(block, title_tags)

            yield Document(docno=docno, full_text=full_text, title_text=title_text)


def parse_topics(topic_file: str) -> list[Topic]:
    """Parse TREC-format topic file (valid XML)."""
    tree = ET.parse(topic_file)
    root = tree.getroot()
    topics = []
    for top in root.findall("top"):
        qid = (top.findtext("num") or "").strip()
        title = (top.findtext("title") or "").strip()
        desc = (top.findtext("desc") or "").strip()
        narr = (top.findtext("narr") or "").strip()
        topics.append(Topic(qid=qid, title=title, desc=desc, narr=narr))
    return topics
