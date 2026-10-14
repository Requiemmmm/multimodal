#!/usr/bin/env python
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.959963984540054):
    if not n: return [float("nan"), float("nan")]
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return [c-h,c+h]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--metrics",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.metrics).read_text().splitlines() if x.strip()]
    groups=defaultdict(list)
    for r in rows: groups[(r["prompt_id"],str(r["alpha"]))].append(r)
    out={"n":len(rows),"groups":{}}
    for (prompt,alpha), xs in sorted(groups.items()):
        k=sum(x["boi_sampled"] for x in xs); out["groups"][f"{prompt}/alpha={alpha}"]={"n":len(xs),"boi_successes":k,"boi_rate":k/len(xs),"wilson95":wilson(k,len(xs)),"mean_step":sum(x["boi_step"] for x in xs if x["boi_step"] is not None)/k if k else None}
    Path(a.output).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
if __name__ == "__main__": main()
