"""
Factual QA benchmark adapter (e.g. SimpleQA format) for correctness evaluation.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FactualQAPair:
    id: str
    question: str
    target_answer: str
    topic: str
    metadata: Dict[str, Any]


class SimpleQABenchmark:
    """
    Standard factual benchmark adapter containing curated factual questions
    and automated exact/semantic evaluation.
    """

    def __init__(self, data: Optional[List[FactualQAPair]] = None):
        self.data: List[FactualQAPair] = data or self._default_sample_benchmark()

    def _default_sample_benchmark(self) -> List[FactualQAPair]:
        return [
            FactualQAPair(
                id="f-001",
                question="What is the capital of France?",
                target_answer="Paris",
                topic="geography",
                metadata={"difficulty": "easy"},
            ),
            FactualQAPair(
                id="f-002",
                question="In what year was the first transformer model (Attention Is All You Need) published?",
                target_answer="2017",
                topic="machine_learning",
                metadata={"difficulty": "medium"},
            ),
            FactualQAPair(
                id="f-003",
                question="What chemical element has the atomic number 6?",
                target_answer="Carbon",
                topic="chemistry",
                metadata={"difficulty": "easy"},
            ),
            FactualQAPair(
                id="f-004",
                question="Who proved Fermat's Last Theorem?",
                target_answer="Andrew Wiles",
                topic="mathematics",
                metadata={"difficulty": "hard"},
            ),
            FactualQAPair(
                id="f-005",
                question="What is the speed of light in vacuum in meters per second (approx)?",
                target_answer="299792458",
                topic="physics",
                metadata={"difficulty": "medium"},
            ),
        ]

    def evaluate_model_outputs(
        self,
        model_predictions: Dict[str, str],
        verifier,
    ) -> Dict[str, Any]:
        """
        Evaluates a set of model predictions against the benchmark ground truth.
        """
        results = []
        correct = 0

        for item in self.data:
            pred = model_predictions.get(item.id, "")
            decision = verifier.verify(pred, context=item.target_answer)
            is_right = decision.is_verified
            if is_right:
                correct += 1

            results.append({
                "id": item.id,
                "question": item.question,
                "prediction": pred,
                "target": item.target_answer,
                "verified": is_right,
                "confidence": decision.confidence,
            })

        acc = correct / len(self.data) if self.data else 0.0
        return {
            "total_items": len(self.data),
            "correct_items": correct,
            "accuracy": acc,
            "item_results": results,
        }
