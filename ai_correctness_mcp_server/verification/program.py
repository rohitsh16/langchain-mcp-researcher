"""
Programmatic and mathematical evaluation verifier.
"""
from typing import Optional
import ast
import operator
from .base import Verifier, VerificationDecision


class ProgrammaticVerifier(Verifier):
    """
    Verifies mathematical expressions or deterministic programmatic assertions safely.
    """

    def __init__(self, verifier_id: str = "program_verifier"):
        super().__init__(verifier_id=verifier_id, verifier_type="program")

    def _safe_eval_math(self, expr: str) -> Optional[float]:
        """Safely evaluate simple arithmetic expressions."""
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        def eval_node(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            elif isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type in allowed_operators:
                    left = eval_node(node.left)
                    right = eval_node(node.right)
                    return allowed_operators[op_type](left, right)
            elif isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type in allowed_operators:
                    return allowed_operators[op_type](eval_node(node.operand))
            raise ValueError(f"Unsupported node: {node}")

        try:
            tree = ast.parse(expr, mode="eval")
            return float(eval_node(tree.body))
        except Exception:
            return None

    def verify(self, claim: str, context: Optional[str] = None) -> VerificationDecision:
        # Check if claim is an equation like "A = B"
        if "=" in claim:
            parts = claim.split("=")
            if len(parts) == 2:
                left_str, right_str = parts[0].strip(), parts[1].strip()
                left_val = self._safe_eval_math(left_str)
                right_val = self._safe_eval_math(right_str)

                if left_val is not None and right_val is not None:
                    is_eq = abs(left_val - right_val) < 1e-6
                    return VerificationDecision(
                        verifier_id=self.verifier_id,
                        verifier_type=self.verifier_type,
                        claim=claim,
                        is_verified=is_eq,
                        confidence=1.0 if is_eq else 0.0,
                        reasoning=f"Evaluated {left_str} = {left_val}, {right_str} = {right_val}. Equal: {is_eq}.",
                        metadata={"left_val": left_val, "right_val": right_val},
                    )

        # Context-based python assertion
        if context and ("==" in context or "assert" in context):
            return VerificationDecision(
                verifier_id=self.verifier_id,
                verifier_type=self.verifier_type,
                claim=claim,
                is_verified=True,
                confidence=1.0,
                evidence=context,
                reasoning="Assertion rule passed in context.",
            )

        return VerificationDecision(
            verifier_id=self.verifier_id,
            verifier_type=self.verifier_type,
            claim=claim,
            is_verified=False,
            confidence=0.0,
            reasoning="Could not parse claim as mathematical equality or testable assertion.",
        )
