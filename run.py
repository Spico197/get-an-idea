import json
from collections import Counter

import stanza
from rich.progress import track
from stanza.models.constituency.parse_tree import Tree


def load_json(filepath):
    with open(filepath, "rt", encoding="utf8") as fin:
        data = json.load(fin)
        return data


class ParseTool:
    def __init__(self) -> None:
        self.nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,constituency")

    def con_parse(self, sent: str) -> Tree:
        doc = self.nlp(sent)
        return doc.sentences[0].constituency

    def get_chunk_in_label(self, tree: Tree, label):
        results = []
        if tree.is_leaf():
            return []
        if tree.label == label:
            results.append(tree.leaf_labels())
        for child in tree.children:
            child_res = self.get_chunk_in_label(child, label)
            if child_res:
                results.extend(child_res)
        return results

    def get_possible_combinations(
        self, string, labels=("NP", "NML")
    ) -> list[tuple[str]]:
        tree = self.con_parse(string.lower())
        combs = []
        for label in labels:
            res = self.get_chunk_in_label(tree, label)
            if res:
                combs.extend([tuple(r) for r in res])
        return combs


def dump_comb_counter(parse_tool: ParseTool, papers: list[dict], desc: str):
    all_combs = []
    for paper in track(papers, desc):
        title = paper["title"]
        combs = parse_tool.get_possible_combinations(title)
        all_combs.extend(combs)

    comb_counter = Counter(all_combs)
    res = comb_counter.most_common()
    with open(f"{desc}.json", "wt", encoding="utf8") as fout:
        json.dump(res, fout, ensure_ascii=False)


if __name__ == "__main__":
    papers = load_json("abs_ie.json")
    parse_tool = ParseTool()

    dump_comb_counter(parse_tool, papers, "outs/all")

    years = sorted(set(p["year"] for p in papers))
    for year in years:
        dump_comb_counter(
            parse_tool,
            list(filter(lambda p: p["year"] == year, papers)),
            f"outs/y{year}",
        )
