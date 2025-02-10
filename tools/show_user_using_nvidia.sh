#!/bin/bash

pids=$(fuser -v /dev/nvidia* | cut -d' ' -f3- | tr ' ' '\n' | sort -u)
for pid in $pids
do
   echo "PID: $pid CWD: $(readlink /proc/$pid/cwd)"
done