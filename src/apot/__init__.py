"""Adaptive-mass mini-batch partial optimal transport: solvers, mass rules, diagnostics."""
from .rules import Batch, build_rule, add_rule_args, rule_name_fields, RULES, LABEL_FREE  # noqa: F401
