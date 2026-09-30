import json
import logging
import urllib.parse
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import httpx

from research.ontology.schemas import PaperRecord

logger = logging.getLogger(__name__)


@runtime_checkable
class PaperIndex(Protocol):
    """Protocol for academic indexing backends."""

    def search(self, query: str, limit: int = 5) -> List[PaperRecord]:
        ...

    def get(self, paper_id: str) -> Optional[PaperRecord]:
        ...


class MockIndex:
    """Deterministic local index for offline development, reproducible testing, and CI."""

    def __init__(self, papers: Optional[List[PaperRecord]] = None):
        if papers is None:
            self.papers = [
                PaperRecord(
                    paper_id="arxiv:2312.11805",
                    title="Gemini: A Family of Highly Capable Multimodal Models",
                    authors=["Gemini Team", "Demis Hassabis", "Jeff Dean"],
                    year=2023,
                    venue="arXiv",
                    doi="10.48550/arXiv.2312.11805",
                    arxiv_id="2312.11805",
                    abstract="This report introduces Gemini, a family of multimodal models trained across text, code, audio, image, and video.",
                    pdf_url="https://arxiv.org/pdf/2312.11805.pdf",
                    source_indexes=["mock_index"],
                ),
                PaperRecord(
                    paper_id="arxiv:2303.08774",
                    title="GPT-4 Technical Report",
                    authors=["OpenAI"],
                    year=2023,
                    venue="arXiv",
                    doi="10.48550/arXiv.2303.08774",
                    arxiv_id="2303.08774",
                    abstract="We report the development of GPT-4, a large-scale, multimodal model which can exhibit human-level performance on benchmarks.",
                    pdf_url="https://arxiv.org/pdf/2303.08774.pdf",
                    source_indexes=["mock_index"],
                ),
                PaperRecord(
                    paper_id="arxiv:2101.02703",
                    title="Conformal Risk Control",
                    authors=["Stephen Bates", "Emmanuel Candes", "Lihua Lei"],
                    year=2021,
                    venue="arXiv",
                    doi="10.48550/arXiv.2101.02703",
                    arxiv_id="2101.02703",
                    abstract="We present a framework for distribution-free risk control that certifies bounds on expected loss under exchangeability.",
                    pdf_url="https://arxiv.org/pdf/2101.02703.pdf",
                    source_indexes=["mock_index"],
                ),
                PaperRecord(
                    paper_id="openalex:W4287819876",
                    title="Conformal Risk Control for Language Models",
                    authors=["Anastasios N. Angelopoulos", "Stephen Bates", "Michael I. Jordan"],
                    year=2024,
                    venue="ICLR",
                    doi="10.48550/arXiv.2306.05262",
                    arxiv_id="2306.05262",
                    abstract="We introduce methods to guarantee that language models satisfy statistical risk bounds for hallucination and correctness.",
                    pdf_url="https://arxiv.org/pdf/2306.05262.pdf",
                    source_indexes=["mock_index"],
                ),
            ]
        else:
            self.papers = papers
        for p in self.papers:
            if not p.fingerprint:
                p.fingerprint = p.compute_fingerprint()

    def search(self, query: str, limit: int = 5) -> List[PaperRecord]:
        q = query.lower()
        results = [
            p
            for p in self.papers
            if q in p.title.lower() or q in (p.abstract or "").lower() or any(q in a.lower() for a in p.authors)
        ]
        if not results:
            results = self.papers
        return results[:limit]

    def get(self, paper_id: str) -> Optional[PaperRecord]:
        for p in self.papers:
            if p.paper_id == paper_id or p.arxiv_id == paper_id:
                return p
        return None


class ArxivIndex:
    """arXiv Academic API index."""

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def search(self, query: str, limit: int = 5) -> List[PaperRecord]:
        import xml.etree.ElementTree as ET

        url = (
            f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}"
            f"&start=0&max_results={limit}&sortBy=relevance&sortOrder=descending"
        )
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url, headers={"User-Agent": "LangChain-MCP-Researcher/0.2.0"})
                if resp.status_code != 200:
                    return []
                root = ET.fromstring(resp.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                records: List[PaperRecord] = []
                for entry in root.findall("atom:entry", ns):
                    raw_id = entry.findtext("atom:id", "", ns)
                    arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
                    title = " ".join((entry.findtext("atom:title", "", ns) or "").split())
                    abstract = " ".join((entry.findtext("atom:summary", "", ns) or "").split())
                    authors = [
                        a.findtext("atom:name", "", ns)
                        for a in entry.findall("atom:author", ns)
                        if a.findtext("atom:name", "", ns)
                    ]
                    published = entry.findtext("atom:published", "", ns)
                    year = int(published[:4]) if published and len(published) >= 4 else None

                    rec = PaperRecord(
                        paper_id=f"arxiv:{arxiv_id}",
                        title=title,
                        authors=authors,
                        year=year,
                        venue="arXiv",
                        arxiv_id=arxiv_id,
                        abstract=abstract,
                        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                        source_indexes=["arxiv"],
                    )
                    rec.fingerprint = rec.compute_fingerprint()
                    records.append(rec)
                return records
        except Exception as e:
            logger.warning("ArxivIndex query failed: %s", e)
            return []

    def get(self, paper_id: str) -> Optional[PaperRecord]:
        clean_id = paper_id.replace("arxiv:", "").strip()
        results = self.search(f"id:{clean_id}", limit=1)
        return results[0] if results else None


class OpenAlexIndex:
    """OpenAlex API index for multidisciplinary scholarly literature."""

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def search(self, query: str, limit: int = 5) -> List[PaperRecord]:
        url = f"https://api.openalex.org/works?search={urllib.parse.quote(query)}&per_page={limit}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url, headers={"User-Agent": "LangChain-MCP-Researcher/0.2.0"})
                if resp.status_code != 200:
                    return []
                data = resp.json()
                records = []
                for item in data.get("results", []):
                    title = item.get("display_name") or "Untitled"
                    authors = [
                        a.get("author", {}).get("display_name")
                        for a in item.get("authorships", [])
                        if a.get("author", {}).get("display_name")
                    ]
                    year = item.get("publication_year")
                    venue = item.get("primary_location", {}).get("source", {}).get("display_name")
                    doi = item.get("doi")
                    openalex_id = item.get("id", "").split("/")[-1]
                    pdf_url = item.get("open_access", {}).get("oa_url")

                    rec = PaperRecord(
                        paper_id=f"openalex:{openalex_id}",
                        title=title,
                        authors=authors,
                        year=year,
                        venue=venue,
                        doi=doi,
                        abstract=None,
                        pdf_url=pdf_url,
                        source_indexes=["openalex"],
                    )
                    rec.fingerprint = rec.compute_fingerprint()
                    records.append(rec)
                return records
        except Exception as e:
            logger.warning("OpenAlexIndex query failed: %s", e)
            return []

    def get(self, paper_id: str) -> Optional[PaperRecord]:
        clean_id = paper_id.replace("openalex:", "").strip()
        url = f"https://api.openalex.org/works/{clean_id}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url, headers={"User-Agent": "LangChain-MCP-Researcher/0.2.0"})
                if resp.status_code != 200:
                    return None
                item = resp.json()
                rec = PaperRecord(
                    paper_id=f"openalex:{clean_id}",
                    title=item.get("display_name") or "Untitled",
                    authors=[
                        a.get("author", {}).get("display_name")
                        for a in item.get("authorships", [])
                        if a.get("author", {}).get("display_name")
                    ],
                    year=item.get("publication_year"),
                    venue=item.get("primary_location", {}).get("source", {}).get("display_name"),
                    doi=item.get("doi"),
                    pdf_url=item.get("open_access", {}).get("oa_url"),
                    source_indexes=["openalex"],
                )
                rec.fingerprint = rec.compute_fingerprint()
                return rec
        except Exception:
            return None


class MultiSourceSearchPlanner:
    """Expands a core research question into a diverse set of search strategies across indexes."""

    def __init__(self, indexes: Optional[List[PaperIndex]] = None, indices: Optional[List[PaperIndex]] = None):
        self.indexes = indexes or indices or [MockIndex()]

    def plan_queries(self, question: str) -> Dict[str, str]:
        """Generates targeted query variants for theory, uncertainty, verification, and formal domains."""
        q = question.strip()
        return {
            "primary": q,
            "theory": f"conformal risk control bounds {q}",
            "uncertainty": f"semantic uncertainty hallucination {q}",
            "verification": f"verifier correctness empirical assessment {q}",
            "formal": f"formal verification certification {q}",
        }

    def execute_plan(self, question: str, limit_per_query: int = 3) -> List[PaperRecord]:
        queries = self.plan_queries(question)
        all_papers: List[PaperRecord] = []
        for category, q_str in queries.items():
            for idx in self.indexes:
                try:
                    papers = idx.search(q_str, limit=limit_per_query)
                    all_papers.extend(papers)
                except Exception as e:
                    logger.warning("Index error on %s: %s", category, e)
        return all_papers

    def search(self, question: str, limit: int = 5) -> List[PaperRecord]:
        return self.execute_plan(question, limit_per_query=limit)
