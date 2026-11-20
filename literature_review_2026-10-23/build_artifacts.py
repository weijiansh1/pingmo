"""Build the readable review, literature matrix, and BibTeX from checked notes."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
records = json.loads((ROOT / "references.json").read_text())

def bib_escape(value):
    return str(value).replace("&", r"\&").replace("%", r"\%")

bib_entries = []
for item in records:
    fields = {
        "title": "{" + item["title"] + "}",
        "author": item["author"],
        "year": item["year"],
        "url": item["url"],
    }
    if item.get("doi"):
        fields["doi"] = item["doi"]
    if item["type"] == "article":
        fields["journal"] = item["venue"]
    elif item["type"] == "inproceedings":
        fields["booktitle"] = item["venue"]
    elif item["type"] == "techreport":
        fields.update(institution="NASA", number="TN D-5153")
    else:
        fields["howpublished"] = item["venue"]
    fields["note"] = f"Literature review ID: {item['id']}"
    body = ",\n".join(f"  {key} = {{{bib_escape(value)}}}" for key, value in fields.items())
    bib_entries.append(f"@{item['type']}{{{item['key']},\n{body}\n}}")
(ROOT / "references.bib").write_text("\n\n".join(bib_entries) + "\n")

columns = ["id", "title", "author", "year", "venue", "group", "summary", "evidence", "read_scope", "note", "doi", "url", "key"]
with (ROOT / "literature_matrix.csv").open("w", newline="", encoding="utf-8-sig") as output:
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)

review = [
    "**飞行品质、飞控基础策略、特权训练与迭代蒸馏：文献梳理**\n",
    (ROOT / "synthesis_zh.md").read_text(),
    "\n以下是全部 32 篇的文献矩阵。每条给出原始来源链接、核心内容、证据对象和本次阅读范围。\n",
]
groups = list(dict.fromkeys(item["group"] for item in records))
for group in groups:
    review.append(f"\n**{group}**\n")
    review.append("| 编号与文献 | 核心内容 | 验证范围 | 本次阅读范围 |\n|---|---|---|---|")
    for item in records:
        if item["group"] != group:
            continue
        title = f"{item['id']} · [{item['title']}]({item['url']}) · {item['year']} · {item['venue']}"
        cells = [title, item["summary"], item["evidence"], item["read_scope"]]
        review.append("| " + " | ".join(value.replace("|", "／").replace("\n", " ") for value in cells) + " |")
    for item in records:
        if item["group"] == group and item.get("note"):
            review.append(f"\n{item['id']} 版本/使用说明：{item['note']}\n")
review.append("\n可编辑数据为 references.json；Excel 可直接打开 literature_matrix.csv；LaTeX 引用使用 references.bib。缓存为本地个人阅读材料，报告的可分享内容是原创摘要及来源链接。\n")
(ROOT / "review_zh.md").write_text("\n".join(review))

assert len(records) == len({item["key"] for item in records})
assert len(records) == len({item["id"] for item in records})
assert all(item["url"].startswith("https://") for item in records)
print(f"Built review, CSV, and BibTeX for {len(records)} references.")
