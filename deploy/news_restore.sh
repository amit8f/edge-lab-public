#!/usr/bin/env bash
# Puts the 2015-10..2025-09 Benzinga headline months back into a news folder when the Actions cache lost them.
# Alpaca stopped serving old news in 2026; the only copy is the encrypted release news-backup-* of the private
# amit8f/edge-lab repo (made by sector1.yml stage backup, Amit 2026-11-21). Amit 2026-11-24: jobs restore from it
# instead of keeping caches warm.
# Usage: bash deploy/news_restore.sh <news dir>   (needs EVIDENCE_TOKEN and ALPACA_API_SECRET_KEY)
set -euo pipefail
dest=${1:?news dir}
mkdir -p "$dest"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
[[ -n "${EVIDENCE_TOKEN:-}" && -n "${ALPACA_API_SECRET_KEY:-}" ]] || { echo "::error::news restore needs EVIDENCE_TOKEN and ALPACA_API_SECRET_KEY"; exit 1; }
tag=$(GH_TOKEN="$EVIDENCE_TOKEN" gh release list --repo amit8f/edge-lab --limit 50 --json tagName,createdAt \
      --jq '[.[] | select(.tagName | startswith("news-backup-"))] | sort_by(.createdAt) | last | .tagName')
[[ -n "$tag" && "$tag" != null ]] || { echo "::error::no news-backup release found"; exit 1; }
GH_TOKEN="$EVIDENCE_TOKEN" gh release download "$tag" --repo amit8f/edge-lab --pattern 'benzinga_news.sha256' --dir "$tmp"
# Already complete? Every backed-up month present with the same bytes.
if ( cd "$dest" && sha256sum --quiet -c "$tmp/benzinga_news.sha256" ) >/dev/null 2>&1; then
  echo "news months already complete in $dest ($(wc -l < "$tmp/benzinga_news.sha256") months, backup $tag)"
  exit 0
fi
GH_TOKEN="$EVIDENCE_TOKEN" gh release download "$tag" --repo amit8f/edge-lab --pattern 'benzinga_news.tar.gz.enc' --dir "$tmp"
openssl enc -d -aes-256-cbc -pbkdf2 -pass env:ALPACA_API_SECRET_KEY -in "$tmp/benzinga_news.tar.gz.enc" | tar -xzf - -C "$tmp"
( cd "$tmp/news" && sha256sum --quiet -c "$tmp/benzinga_news.sha256" ) || { echo "::error::backup failed its checksum"; exit 1; }
# The backup is the reference copy of these months: it replaces whatever the cache holds for them (a fresh
# fetch now writes empty months), and leaves later months alone.
cp -f "$tmp"/news/news_*.parquet "$dest"/
echo "restored $(ls "$tmp"/news/news_*.parquet | wc -l) news months into $dest from $tag"
