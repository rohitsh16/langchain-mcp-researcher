"""
Unit tests for multi-index search planner and deduplication.
"""
import pytest
from research.literature.search import MultiSourceSearchPlanner, MockIndex
from research.literature.dedup import deduplicate_papers
from research.ontology.schemas import PaperRecord


def test_mock_search_planner():
    mock_papers = [
        PaperRecord(
            paper_id="p-001",
            title="Conformal Risk Control",
            authors=["Stephen Bates", "Anastasios Angelopoulos"],
            year=2021,
            doi="10.1234/crc.2021",
        ),
        PaperRecord(
            paper_id="p-002",
            title="Semantic Uncertainty in Large Language Models",
            authors=["Lorenz Kuhn", "Yarin Gal"],
            year=2023,
            arxiv_id="2302.09664",
        ),
    ]
    planner = MultiSourceSearchPlanner(indices=[MockIndex(mock_papers)])
    results = planner.search("conformal risk control", limit=5)
    assert len(results) >= 1
    assert results[0].title == "Conformal Risk Control"


def test_paper_deduplication():
    papers = [
        PaperRecord(
            paper_id="p-001",
            title="Conformal Risk Control",
            authors=["Bates", "Angelopoulos"],
            year=2021,
            doi="10.1234/crc.2021",
        ),
        PaperRecord(
            paper_id="p-002",
            title="Conformal Risk Control (arXiv Preprint)",
            authors=["Bates", "Angelopoulos"],
            year=2021,
            doi="10.1234/crc.2021",  # Same DOI!
        ),
        PaperRecord(
            paper_id="p-003",
            title="A completely different paper",
            authors=["Smith"],
            year=2024,
        ),
    ]
    deduped = deduplicate_papers(papers)
    assert len(deduped) == 2
    assert len([p for p in deduped if "Conformal" in p.title]) == 1
