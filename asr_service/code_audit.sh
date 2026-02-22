#!/bin/bash
# Code audit script for ASR service
# Runs various linting and analysis tools to check code quality

echo "=== RUFF CHECK ===" > audit_report.txt
uv run --with-editable . --group dev ruff check . >> audit_report.txt 2>&1

echo -e "\n=== BANDIT SECURITY ===" >> audit_report.txt
uv run --with-editable . --group dev bandit -r src >> audit_report.txt 2>&1

echo -e "\n=== MYPY TYPE CHECK ===" >> audit_report.txt
uv run --with-editable . --group dev mypy src >> audit_report.txt 2>&1

echo -e "\n=== RADON COMPLEXITY ===" >> audit_report.txt
uv run --with-editable . --group dev radon cc src -a >> audit_report.txt 2>&1

echo -e "\n=== VULTURE DEAD CODE ===" >> audit_report.txt
uv run --with-editable . --group dev vulture src >> audit_report.txt 2>&1

echo -e "\n=== PYLINT ===" >> audit_report.txt
uv run --with-editable . --group dev pylint src >> audit_report.txt 2>&1

# Copy the final report to clipboard
cat audit_report.txt | pbcopy
echo "Done! Full report saved to 'audit_report.txt' and copied to clipboard."