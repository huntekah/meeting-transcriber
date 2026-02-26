#!/bin/bash
# Code audit script for ASR service
# Runs various linting and analysis tools to check code quality

print_summary() {
  echo ""
  echo "=== AUDIT SUMMARY ==="
  awk '/^=== RUFF CHECK ===/,/^=== BANDIT/ { if (/^Found.*error/) print "RUFF: " $0 }' audit_report.txt | head -1
  awk '/^=== BANDIT SECURITY ===/,/^=== MYPY/ { if (/^>>/) print "BANDIT: " $0 }' audit_report.txt | head -1
  awk '/^=== MYPY TYPE CHECK ===/,/^=== RADON/ { if (/^Found.*error/) print "MYPY: " $0 }' audit_report.txt | head -1
  radon_output=$(awk '/^=== RADON COMPLEXITY ===/,/^=== VULTURE/ { print }' audit_report.txt | grep -E "^\s+[A-Z]" | head -1)
  [ -n "$radon_output" ] && echo "RADON: $radon_output" || echo "RADON: ✓ No high complexity functions"
  vulture_count=$(awk '/^=== VULTURE DEAD CODE ===/,/^=== PYLINT/ { if (/unused/) count++ } END { print count+0 }' audit_report.txt)
  if [ "$vulture_count" -gt 0 ]; then echo "VULTURE: $vulture_count unused items found"; else echo "VULTURE: ✓ No unused code"; fi
  pylint_score=$(awk '/^=== PYLINT ===/,/^=== END/ { if (/rated at/) print }' audit_report.txt | tail -1)
  if [ -n "$pylint_score" ]; then
    echo "PYLINT: $pylint_score"
  else
    echo "PYLINT: (score not found)"
  fi
}

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
echo -e "\n=== END ===" >> audit_report.txt

# Copy the final report to clipboard
cat audit_report.txt | pbcopy

# Print summary to console
print_summary

echo "Done! Full report saved to 'audit_report.txt' and copied to clipboard."
