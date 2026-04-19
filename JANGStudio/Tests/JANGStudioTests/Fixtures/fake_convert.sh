#!/bin/bash
# Emits 5 phase events, 10 tick events, and a done:true event on stderr.
# Emits banner text on stdout.
set -e
echo "JANG Convert v2 — fake"          # stdout
for i in 1 2 3 4 5; do
  echo "{\"v\":1,\"type\":\"phase\",\"n\":$i,\"total\":5,\"name\":\"p$i\",\"ts\":1.0}" >&2
  sleep 0.02
done
for i in $(seq 0 9); do
  echo "{\"v\":1,\"type\":\"tick\",\"done\":$i,\"total\":10,\"label\":\"t$i\",\"ts\":1.0}" >&2
done
echo "{\"v\":1,\"type\":\"done\",\"ok\":true,\"output\":\"/tmp/out\",\"elapsed_s\":0.5,\"ts\":2.0}" >&2
