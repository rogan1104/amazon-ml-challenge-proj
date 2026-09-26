"""Inspect typical train name/address blocking bucket sizes on a row sample."""
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, r"student_resource/code/business_entity_resolution/src")
from normalize import normalize_text

SKIP = frozenset({"private", "limited", "incorporated", "corporation", "llc", "llp", "plc", "company", "and", "the", "of", "dba", "india", "services", "service", "group", "holdings", "holding"})


def keys(country, name, addr):
    name_n, addr_n = normalize_text(name), normalize_text(addr)
    out = []
    tokens = [t for t in name_n.split() if len(t) >= 4 and t not in SKIP]
    for token in tokens[:3]:
        out.append((country, "np", token[:4]))
    for digits in re.findall(r"\d{3,}", addr_n)[:2]:
        out.append((country, "dg", digits))
    if not out:
        out.append((country, "fb", name_n[:3] if name_n else "_"))
    return out


base = r"student_resource/dataset/train"
s1 = pd.read_csv(base + "/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=80000)
s2 = pd.read_csv(base + "/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=80000)
counter = Counter()
for frame in (s1, s2):
    for country, name, address in zip(frame["country"], frame["business_name"], frame["business_address"]):
        for key in set(keys(country, name, address)):
            counter[key] += 1
sizes = sorted(counter.values(), reverse=True)
print("n keys", len(counter), "max", sizes[0], "p50", sizes[len(sizes)//2], "p95", sizes[int(len(sizes)*0.05)], "gt10k", sum(s > 10000 for s in sizes), "gt3k", sum(s > 3000 for s in sizes))
print("top 15", counter.most_common(15))
