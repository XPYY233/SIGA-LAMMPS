"""SIGA-LAMMPS adapter core (layer 3).

M  adapter/memory/      the always-on primer (a document)
R  adapter/retrieval/   the LAMMPS knowledge base
X  adapter/validator/   deterministic rules, no LLM

`adapter/cli.py` is the single entry point every consumer shares — the MCP
server's tools, the harness stop gate, and the benchmark evaluator — so X and S
provably apply the same checks.
"""
