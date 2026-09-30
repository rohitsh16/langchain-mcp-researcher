"""
Multi-source scholarly paper deduplication.
Merges candidate paper records across arXiv, OpenAlex, and Crossref into unified canonical records.
"""
from typing import Dict, List
from research.ontology.schemas import PaperRecord


class PaperDeduplicator:
    @staticmethod
    def deduplicate(papers: List[PaperRecord]) -> List[PaperRecord]:
        """
        Deduplicates paper records by DOI, arXiv ID, or title-author fingerprint.
        Preserves provenance by combining source_indexes and retaining richest metadata.
        """
        by_doi: Dict[str, PaperRecord] = {}
        by_arxiv: Dict[str, PaperRecord] = {}
        by_fingerprint: Dict[str, PaperRecord] = {}

        unique_papers: List[PaperRecord] = []

        for p in papers:
            if not p.fingerprint:
                p.fingerprint = p.compute_fingerprint()

            match: PaperRecord = None

            if p.doi:
                norm_doi = p.doi.lower().strip()
                if norm_doi in by_doi:
                    match = by_doi[norm_doi]

            if match is None and p.arxiv_id:
                norm_arxiv = p.arxiv_id.strip()
                if norm_arxiv in by_arxiv:
                    match = by_arxiv[norm_arxiv]

            if match is None and p.fingerprint in by_fingerprint:
                match = by_fingerprint[p.fingerprint]

            if match is not None:
                # Merge into match
                for src in p.source_indexes:
                    if src not in match.source_indexes:
                        match.source_indexes.append(src)
                if not match.abstract and p.abstract:
                    match.abstract = p.abstract
                if not match.pdf_url and p.pdf_url:
                    match.pdf_url = p.pdf_url
                if not match.doi and p.doi:
                    match.doi = p.doi
                if not match.arxiv_id and p.arxiv_id:
                    match.arxiv_id = p.arxiv_id
                if not match.year and p.year:
                    match.year = p.year
            else:
                if p.doi:
                    by_doi[p.doi.lower().strip()] = p
                if p.arxiv_id:
                    by_arxiv[p.arxiv_id.strip()] = p
                by_fingerprint[p.fingerprint] = p
                unique_papers.append(p)

        return unique_papers


deduplicate_papers = PaperDeduplicator.deduplicate
