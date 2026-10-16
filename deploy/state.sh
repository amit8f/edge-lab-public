#!/usr/bin/env bash
# Restore or save the orphan branch `state` in ./state (the forward data written by the jobs).
# restore: clone the branch, or start an empty orphan branch on the first run.
# save:    commit everything under ./state and push it back (no-op when nothing changed).
# The token is passed per command and never written to disk.
set -euo pipefail
mode="$1"
repo_url="${STATE_REMOTE:-https://github.com/${GITHUB_REPOSITORY:-}.git}"
branch="${STATE_BRANCH:-state}"   # the crypto track keeps its own branch so hourly runs never race the others
auth=(-c "http.extraheader=AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)")

if [[ "$mode" == "restore" ]]; then
  if git "${auth[@]}" ls-remote --exit-code --heads "$repo_url" "$branch" >/dev/null 2>&1; then
    git "${auth[@]}" clone -q --depth 1 --branch "$branch" "$repo_url" state
  else
    mkdir -p state && cd state && git init -q -b "$branch" && git remote add origin "$repo_url"
    echo "forward data written by the GitHub Actions jobs; never edit by hand" > README.md
  fi
elif [[ "$mode" == "save" ]]; then
  [[ -d state ]] || { echo "no state dir"; exit 0; }
  cd state
  git config user.name "github-actions[bot]"
  git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  git add -A
  if git diff --cached --quiet; then
    echo "no state change"
  else
    git commit -q -m "state: ${RUN_LABEL:-manual}"
    git "${auth[@]}" push -q origin "HEAD:$branch"
    git log --oneline -1
  fi
else
  echo "usage: state.sh restore|save" >&2; exit 2
fi
