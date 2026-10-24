#!/usr/bin/env bash
# run_vllm_tp.sh 를 감시하며 실행한다. 어느 노드든 tbv 재전송/active timeout/커널 WARNING 이 찍히면
# 즉시 양쪽 tbvbench 컨테이너를 죽이고 중단한다(멈춘 QP 를 오래 붙잡으면 재연결 때 커널 손상 위험).
#   사용: [환경변수...] ./run_guarded.sh <tp> <batch> <out>
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
B_HOST="${B_HOST:-parks-zbook@10.99.0.2}"
PAT='native retransmit|active timeout|WARNING:|list_del|blocked for more'
since=$(date '+%Y-%m-%d %H:%M:%S')
INNER_TO="${INNER_TO:-420}" "$D/run_vllm_tp.sh" "$@" > /tmp/.guard_run 2>&1 &
RUN=$!
trip=""
while kill -0 $RUN 2>/dev/null; do
  a=$(journalctl -k --since "$since" --no-pager 2>/dev/null | grep -E "$PAT" | head -3)
  b=$(ssh -o ConnectTimeout=5 "$B_HOST" "journalctl -k --since '$since' --no-pager | grep -E '$PAT' | head -3" 2>/dev/null)
  if [ -n "$a$b" ]; then trip="A:${a:+$a}
B:${b:+$b}"; break; fi
  sleep 3
done
if [ -n "$trip" ]; then
  echo "### 감시 중단 $(date +%T): 커널 이상 감지"; echo "$trip" | cut -c1-220
  for c in $(docker ps -q --filter label=tbvbench=1); do docker kill "$c" >/dev/null; done
  ssh "$B_HOST" 'for c in $(docker ps -q --filter label=tbvbench=1); do docker kill $c >/dev/null; done'
  wait $RUN 2>/dev/null
fi
grep -E "출력해시|결과:|tbvcoll: 호출|워치독|Error|rc=" /tmp/.guard_run | head -12
echo "--- 커널 로그(시작 이후, 양 노드 이상 건수): A=$(journalctl -k --since "$since" --no-pager | grep -cE "$PAT") B=$(ssh "$B_HOST" "journalctl -k --since '$since' --no-pager | grep -cE '$PAT'")"
echo "--- 잔여 컨테이너: A=$(docker ps -q --filter label=tbvbench=1 | wc -l) B=$(ssh "$B_HOST" 'docker ps -q --filter label=tbvbench=1 | wc -l')"
