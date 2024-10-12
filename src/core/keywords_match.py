import re
import json
import zipfile
import pathlib
from datetime import datetime
from collections import defaultdict

from tqdm import tqdm


def load_file_iter(filepath):
    # download from https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download

    if filepath.endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            with zip_ref.open("arxiv-metadata-oai-snapshot.json", "r") as json_file:
                for line in json_file:
                    yield json.loads(line)
    else:
        with open(filepath, "r", encoding="utf8") as fin:
            for line in fin:
                yield json.loads(line)

    # data = []
    # with open("arxiv-metadata-oai-snapshot.json", "r", encoding="utf8") as fin:
    #     for line in fin:
    #         data.append(json.loads(line))
    # return data


def dump_json(data, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2)


def contains_any(string: str, strs: list) -> bool:
    if any(sub.lower() in string.lower() for sub in strs):
        return True
    return False


def parse_date(string):
    string = string.replace("-99-", "-01-")
    return datetime.strptime(string, "%Y-%m-%d")


space_regex = re.compile(r"\s+")


def clean_ins(ins):
    ins["title"] = space_regex.sub(" ", ins["title"].strip())
    ins["abstract"] = space_regex.sub(" ", ins["abstract"].strip())
    return ins


if __name__ == "__main__":
    # unzip -p archive.zip arxiv-metadata-oai-snapshot.json | wc -l
    # fmt: off
    start_date = "1900-11-25"
    # start_date = "2019-01-01"

    # n_tot = 2564718  # arxiv old
    n_tot = 2575090  # arxiv 20241006
    data_type = "arxiv"
    data_path = "data/archive-20241006.zip"

    # n_tot = 100757  # acl 20241008
    # data_type = "acl"
    # data_path = "anthology+abstracts-20241008.json"

    kws_dict = {
        "moe": ["mixture-of-experts", " moe ", "mixture of experts"],
        "llm": ["large language model", " llm ", " llms ", " vlm ", " vlms ", "vision language model", "vision-language model"],
        "safety": [" safe", "refusal", "adversarial attack"],
        "rlhf": ["reinforcement learning", "feedback", " rlhf ", "alignment"],
        "docee": ["document-level event extraction"],
        "ee": ["event extraction"],
        "ie": ["information extraction"],
        "gui": [" gui agent", " gui "],
        "tool": ["tool learning", "function calling", "tool calling"],
        "cross_domain": ["cross-domain", "cross domain"],
        "ner": ["named entity recognition", " ner "],
        "srl": ["semantic role labeling"],
        "task_planning": [" task planning "],
    }
    special_joints = {
        "llm_moe": ["llm", "moe"],
        "llm_rlhf": ["llm", "rlhf"],
        "llm_safety": ["llm", "safety"],
        "llm_task_planning": ["llm", "task_planning"],
        "llm_moe_rlhf": ["llm", "moe", "rlhf"],
        "cross_domain_ner": ["cross_domain", "ner"],
        "cross_domain_srl": ["cross_domain", "srl"],
    }

    categories = ["cs.AI", "cs.CL", "cs.CV"]
    # fmt: on

    def get_type(ins: dict) -> list:
        # return "moe", "rlhf", "llm", "llm_moe", "llm_rlhf", "llm_moe_rlhf"
        if not any(cat in ins["categories"] for cat in categories):
            return []
        if parse_date(ins["update_date"]) < parse_date(start_date):
            return []
        title = ins["title"].lower()
        abstract = ins["abstract"].lower()
        ins_types = set()
        for topic, kws in kws_dict.items():
            if contains_any(title, kws) or contains_any(abstract, kws):
                ins_types.add(topic)

        for joint, kws in special_joints.items():
            if all(ins_type in ins_types for ins_type in kws):
                ins_types.add(joint)
            # print(f"joint, kws: {joint}, {kws}")
            # if all(contains_any(title, kw) or contains_any(abstract, kw) for kw in kws):
            #     ins_types.append(joint)

        # # topics with llm
        # if "llm" in ins_types:
        #     for ins_type in ins_types:
        #         if ins_type != "llm":
        #             ins_types.append(f"llm_{ins_type}")

        return list(ins_types)

    type2data = defaultdict(set)
    id2data = {}
    all_data = load_file_iter(data_path)
    for ins_idx, ins in enumerate(tqdm(all_data, total=n_tot, ncols=80)):
        ins = clean_ins(ins)
        ins_types = get_type(ins)
        if ins_types:
            id2data[ins["id"]] = ins
            for ins_type in ins_types:
                type2data[ins_type].add((ins["id"], parse_date(ins["update_date"])))

    output_dir = pathlib.Path(f"results/{data_type}/{start_date}")
    output_dir.mkdir(exist_ok=True, parents=True)
    final_info = ""
    for ins_type, index_data in tqdm(type2data.items(), total=len(type2data), ncols=80):
        info = f"{ins_type}: {len(index_data)}"
        print(info)
        final_info += info + "\n"
        index_data = sorted(list(index_data), key=lambda x: x[1], reverse=True)
        dump_data = [id2data[x[0]] for x in index_data]
        dump_json(dump_data, output_dir / f"{ins_type}.json")

    with open(output_dir / "info.txt", "w") as f:
        f.write(final_info)
