"""
Full-paper evidence extraction and span segmentation.
Preserves page and section provenance for theorems, assumptions, claims, and experimental evidence.
"""
import hashlib
import json
import os
import re
from typing import Dict, List, Optional

from research.ontology.schemas import EvidenceSpan, EvidenceType


class EvidenceStore:
    """Manages indexed evidence spans with provenance preservation."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "data", "evidence")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._spans: Dict[str, EvidenceSpan] = {}

    def add_span(self, span: EvidenceSpan):
        self._spans[span.evidence_id] = span

    def get_span(self, evidence_id: str) -> Optional[EvidenceSpan]:
        return self._spans.get(evidence_id)

    def get_by_paper(self, paper_id: str) -> List[EvidenceSpan]:
        return [s for s in self._spans.values() if s.paper_id == paper_id]

    def search_evidence(self, query: str, limit: int = 5) -> List[EvidenceSpan]:
        q = query.lower()
        matches = [s for s in self._spans.values() if q in s.text.lower() or (s.section and q in s.section.lower())]
        return matches[:limit]


class PDFEvidenceExtractor:
    """Extracts typed evidence spans (theorems, definitions, claims, assumptions) from text or PDF sections."""

    PATTERNS = {
        EvidenceType.THEOREM: re.compile(
            r"(?i)(?:^|\n)\s*(?:Theorem|Proposition|Corollary)\s+(\d+[\.\d]*)\s*[:\.]?\s*([\s\S]*?)(?=(?:\n\s*(?:Proof|Lemma|Theorem|Definition|\d+\.)|\z))"
        ),
        EvidenceType.LEMMA: re.compile(
            r"(?i)(?:^|\n)\s*Lemma\s+(\d+[\.\d]*)\s*[:\.]?\s*([\s\S]*?)(?=(?:\n\s*(?:Proof|Lemma|Theorem|Definition|\d+\.)|\z))"
        ),
        EvidenceType.DEFINITION: re.compile(
            r"(?i)(?:^|\n)\s*Definition\s+(\d+[\.\d]*)\s*[:\.]?\s*([\s\S]*?)(?=(?:\n\s*(?:Proof|Lemma|Theorem|Definition|\d+\.)|\z))"
        ),
        EvidenceType.PROOF: re.compile(
            r"(?i)(?:^|\n)\s*Proof[\.:]?\s*([\s\S]*?)(?=(?:\n\s*(?:Q\.E\.D\.|\u25a0|\u25a1|Theorem|Lemma|Definition|\d+\.)|\z))"
        ),
        EvidenceType.LIMITATION: re.compile(
            r"(?i)(?:^|\n)\s*(?:\d+\.?\s*)?Limitations?[:\s\n]+([\s\S]*?)(?=(?:\n\s*(?:Conclusion|References|\d+\.)|\z))"
        ),
    }

    @classmethod
    def extract_from_text(cls, text: str, paper_id: str, default_page: int = 1) -> List[EvidenceSpan]:
        spans: List[EvidenceSpan] = []

        # 1. Structural Regex Patterns (Theorems, Proofs, Definitions)
        for ev_type, pattern in cls.PATTERNS.items():
            for match in pattern.finditer(text):
                matched_text = match.group(0).strip()
                if len(matched_text) < 15:
                    continue
                h = hashlib.sha256(matched_text.encode("utf-8")).hexdigest()[:8]
                span_id = f"ev:{paper_id}:{ev_type.value}:{h}"
                spans.append(
                    EvidenceSpan(
                        evidence_id=span_id,
                        paper_id=paper_id,
                        page=default_page,
                        section=ev_type.value.capitalize(),
                        start_offset=match.start(),
                        end_offset=match.end(),
                        text=matched_text[:2000],
                        evidence_type=ev_type,
                    )
                )

        # 2. General Paragraph Spans
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 60]
        for i, para in enumerate(paragraphs):
            h = hashlib.sha256(para.encode("utf-8")).hexdigest()[:8]
            span_id = f"ev:{paper_id}:p{i}:{h}"
            ev_type = EvidenceType.CLAIM
            if "assume" in para.lower() or "assumption" in para.lower():
                ev_type = EvidenceType.CLAIM
            elif "we find that" in para.lower() or "results show" in para.lower():
                ev_type = EvidenceType.RESULT

            spans.append(
                EvidenceSpan(
                    evidence_id=span_id,
                    paper_id=paper_id,
                    page=default_page,
                    section=f"Paragraph {i+1}",
                    text=para[:1500],
                    evidence_type=ev_type,
                )
            )

        return spans
