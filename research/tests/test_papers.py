"""
Unit tests for full-paper evidence extraction and span indexing.
"""
import pytest
import tempfile
import os
from research.papers.evidence import PDFEvidenceExtractor, EvidenceStore
from research.ontology.schemas import EvidenceType


def test_pdf_evidence_extractor_theorems_and_definitions():
    paper_text = """
    1. Introduction
    We establish finite-sample guarantees for output calibration.

    Definition 1: A conformal predictor C maps an input X to a subset of labels.

    Theorem 1: Under the assumption of exchangeability, the coverage probability of C is bounded:
    P(Y in C(X)) >= 1 - alpha.

    Proof: By exchangeability, the ranks of nonconformity scores are uniformly distributed. Q.E.D.

    Limitations:
    Our results assume exchangeable calibration and test distributions.
    """
    spans = PDFEvidenceExtractor.extract_from_text(paper_text, paper_id="arxiv:2302.00001", default_page=3)

    types_found = {s.evidence_type for s in spans}
    assert EvidenceType.DEFINITION in types_found
    assert EvidenceType.THEOREM in types_found
    assert EvidenceType.PROOF in types_found

    # Check provenance fields
    for s in spans:
        assert s.paper_id == "arxiv:2302.00001"
        assert s.page == 3
        assert s.evidence_id.startswith("ev:")


def test_evidence_store_index_and_search():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = EvidenceStore(cache_dir=tmp_dir)
        paper_text = "Theorem 2: The empirical risk converges at rate O(1 / sqrt(n))."
        spans = PDFEvidenceExtractor.extract_from_text(paper_text, paper_id="p-100", default_page=1)
        for s in spans:
            store.add_span(s)

        matched = store.search_evidence("empirical risk", limit=5)
        assert len(matched) >= 1
        assert matched[0].paper_id == "p-100"
        assert "empirical risk" in matched[0].text.lower()
