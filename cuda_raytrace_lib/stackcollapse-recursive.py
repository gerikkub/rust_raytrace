#!/usr/bin/env python3

import sys;

for line in sys.stdin:
    stack, count = line.rsplit(' ', 1)
    parts_set = {}
    for part in stack.split(';'):
        parts_set[part] = None
    
    print(';'.join(parts_set.keys()) + f" {count}")