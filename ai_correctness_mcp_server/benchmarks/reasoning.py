"""
Reasoning and multi-step math benchmark adapter (GSM8K/MATH style).
"""
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass


@dataclass
class ReasoningProblem:
    id: str
    problem: str
    solution_steps: List[str]
    final_answer: str


class MathReasoningBenchmark:
    """
    Adapter for multi-step reasoning evaluation.
    Checks both intermediate step validity and final extracted numeric answers.
    """

    def __init__(self, problems: Optional[List[ReasoningProblem]] = None):
        self.problems = problems or self._default_problems()

    def _default_problems(self) -> List[ReasoningProblem]:
        return [
            ReasoningProblem(
                id="r-001",
                problem="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether?",
                solution_steps=[
                    "Natalia sold 48 clips in April.",
                    "In May, she sold 48 / 2 = 24 clips.",
                    "Altogether, she sold 48 + 24 = 72 clips.",
                ],
                final_answer="72",
            ),
            ReasoningProblem(
                id="r-002",
                problem="Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
                solution_steps=[
                    "Weng earns $12 for 60 minutes.",
                    "She earns 12 / 60 = $0.20 per minute.",
                    "For 50 minutes, she earns 50 * 0.20 = $10.",
                ],
                final_answer="10",
            ),
        ]

    def extract_final_number(self, text: str) -> Optional[str]:
        # Search for boxed or last number in text
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        return numbers[-1] if numbers else None

    def evaluate_solution(self, problem: ReasoningProblem, generated_text: str) -> Dict[str, Any]:
        extracted = self.extract_final_number(generated_text)
        target = self.extract_final_number(problem.final_answer)

        is_correct = (extracted == target) if (extracted and target) else False

        # Check step overlap
        steps_grounded = 0
        gen_lower = generated_text.lower()
        for step in problem.solution_steps:
            key_tokens = [w for w in step.lower().split() if len(w) > 3]
            match_count = sum(1 for w in key_tokens if w in gen_lower)
            if key_tokens and (match_count / len(key_tokens) >= 0.5):
                steps_grounded += 1

        process_score = steps_grounded / len(problem.solution_steps) if problem.solution_steps else 0.0

        return {
            "problem_id": problem.id,
            "extracted_answer": extracted,
            "target_answer": target,
            "final_answer_correct": is_correct,
            "process_score": process_score,
            "fully_correct": is_correct and (process_score >= 0.5),
        }
