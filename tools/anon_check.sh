#!/usr/bin/env bash
# Anonymization gate for this release. Run before every push.
# 1) git metadata: every commit's author AND committer must be the Anonymous identity
#    (this leg exists because three commits once reached the remote with a real
#    identity and had to be history-rewritten, 2026-09-24).
# 2) content: no internal hosts/identities in tracked files.
set -u
fail=0
bad=$(git log --all --format='%h %an <%ae> %cn <%ce>' | grep -v '^\S* Anonymous <anon@example.com> Anonymous <anon@example.com>$' || true)
if [ -n "$bad" ]; then echo "ANON-CHECK FAIL (git metadata):"; echo "$bad"; fail=1; fi
hits=$(git grep -I -n -E '10\.2\.0\.[0-9]+|markruss|russinovich|ahmsalem|azureuser@|MLTraining' -- . 2>/dev/null | grep -v "tools/anon_check.sh" || true)
if [ -n "$hits" ]; then echo "ANON-CHECK FAIL (content):"; echo "$hits" | head -20; fail=1; fi
[ $fail -eq 0 ] && echo "ANON-CHECK OK (git metadata + content)"
exit $fail
