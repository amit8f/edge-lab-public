#!/usr/bin/env bash
# Restore or save the orphan branch `state` in ./state (the forward data written by the jobs).
# STATE_DIR names another folder (SYS2B keeps the private evidence clone in ./evidence).
# restore: clone the branch, or start an empty orphan branch on the first run.
# save:    commit everything under ./state and push it back (no-op when nothing changed).
# The token is passed per command and never written to disk.
set -euo pipefail
mode="$1"
repo_url="${STATE_REMOTE:-https://github.com/${GITHUB_REPOSITORY:-}.git}"
dir="${STATE_DIR:-state}"
branch="${STATE_BRANCH:-state}"   # the crypto track keeps its own branch so hourly runs never race the others
auth=(-c "http.extraheader=AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)")

if [[ "$mode" == "restore" ]]; then
  if git "${auth[@]}" ls-remote --exit-code --heads "$repo_url" "$branch" >/dev/null 2>&1; then
    git "${auth[@]}" clone -q --depth 1 --branch "$branch" "$repo_url" "$dir"
  else
    mkdir -p "$dir" && cd "$dir" && git init -q -b "$branch" && git remote add origin "$repo_url"
    echo "forward data written by the GitHub Actions jobs; never edit by hand" > README.md
  fi
elif [[ "$mode" == "save" ]]; then
  [[ -d "$dir" ]] || { echo "no $dir dir"; exit 0; }
  cd "$dir"
  git config user.name "github-actions[bot]"
  git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  git add -A
  if git diff --cached --quiet; then
    echo "no state change"
  else
    git commit -q -m "state: ${RUN_LABEL:-manual}"
    # Bots can save at the same time (each has its own concurrency group). Rebase onto the newer state and retry.
    pushed=0
    for try in 1 2 3 4; do
      if git "${auth[@]}" push -q origin "HEAD:$branch"; then pushed=1; break; fi
      sleep $((try * 2))
      git "${auth[@]}" fetch -q origin "$branch"
      git rebase -q FETCH_HEAD
    done
    [[ "$pushed" == 1 ]] || { echo "could not push the state after 4 tries" >&2; exit 1; }
    git log --oneline -1
  fi
else
  echo "usage: state.sh restore|save" >&2; exit 2
fi
