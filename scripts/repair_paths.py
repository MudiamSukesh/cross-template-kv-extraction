import pathlib
OLD = r"cross-template-form-extraction\data\dataset"
NEW = r"finalproject\data"
pairs = [(OLD.replace("\\", "\\\\"), NEW.replace("\\", "\\\\")),
         (OLD.replace("\\", "/"),    NEW.replace("\\", "/"))]
for name in ("cache/docs.jsonl", "cache/vrdu.jsonl", "cache/vrdu_doctr.jsonl"):
    p = pathlib.Path(name)
    if not p.exists():
        continue
    tmp, n = p.with_suffix(".tmp"), 0
    with p.open("r", encoding="utf-8") as fi, tmp.open("w", encoding="utf-8") as fo:
        for line in fi:
            if any(o in line for o, _ in pairs):
                n += 1
            for o, w in pairs:
                line = line.replace(o, w)
            fo.write(line)
    tmp.replace(p)
    print(f"{name}: {n} lines rewritten")
