#!/usr/bin/env bash
# Build the code zip for Canvas: source, tests, data, docs and (optionally) your signed contribution sheet.
#
#   scripts/make_submission.sh [path/to/signed_contribution_sheet.pdf]
#
# Excludes: .git, results/, caches, the design-notes PDF. Run `python -m pytest` first.
set -euo pipefail
cd "$(dirname "$0")/.."
out="psoann_submission.zip"
rm -f "$out"
files=$(git ls-files | grep -v -E '^(repo-design\.md\.pdf)$' || true)
# also include untracked-but-not-ignored files (new work not yet committed)
files="$files
$(git ls-files --others --exclude-standard)"
echo "$files" | sort -u | grep -v '^$' | zip -q "$out" -@
if [ "${1:-}" != "" ]; then
  [ -f "$1" ] || { echo "contribution sheet not found: $1" >&2; exit 1; }
  zip -q -j "$out" "$1"
else
  echo "WARNING: no signed contribution sheet given. No marks are issued without it." >&2
fi
echo "wrote $out ($(unzip -l "$out" | tail -1 | awk '{print $2}') files)"
echo "check the clean-checkout run:  unzip -d /tmp/check $out && cd /tmp/check && pip install -e '.[dev]' && python -m pytest && psoann run --runs 3"
