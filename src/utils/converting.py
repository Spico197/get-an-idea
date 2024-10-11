import re
import json
import gzip
import pathlib

from tqdm import tqdm


def dump_jsonlines(data, filepath):
    with open(filepath, "wt", encoding="utf8") as fout:
        for ins in data:
            fout.write(json.dumps(ins, ensure_ascii=False) + "\n")


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def remove_suffix(text: str, suffix: str) -> str:
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def remove_prefix_suffix(text: str, prefix: str, suffix: str) -> str:
    return remove_suffix(remove_prefix(text, prefix), suffix)


# fmt: off
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
# fmt: one


def parse_bib_month(month: str) -> int:
    if month.isdigit():
        return int(month)
    elif month.lower() in MONTH_MAP:
        return MONTH_MAP[month.lower()]
    else:
        return 1


def parse_bib(input_filepath: str, output_filepath: str) -> list:
    input_filepath = pathlib.Path(input_filepath)
    output_filepath = pathlib.Path(output_filepath)
    if input_filepath.suffix == ".gz":
        open_func = gzip.open
    else:
        open_func = open

    data = []
    with open_func(input_filepath, "rt", encoding="utf8") as fin:
        tot_bib_string = fin.read()
        tot_bib_string = re.sub(
            r"  and\n\s+", "  and  ", tot_bib_string, flags=re.MULTILINE
        )
        tot_entries = tot_bib_string.count("@")
        for bib in tqdm(
            re.finditer(
                r"@(\w+)\{(.+?),\n(.*?)\}$",
                tot_bib_string,
                flags=re.MULTILINE | re.DOTALL,
            ),
            desc="parse bib",
            total=tot_entries,
        ):
            bib_type = bib.group(1)
            bib_key = bib.group(2)
            bib_content = {}
            content_string = bib.group(3).strip()
            for val in re.finditer(
                r"\s*(.*?)\s*=\s*(.+?),$\n", content_string, flags=re.MULTILINE
            ):
                bib_content[val.group(1).strip()] = (
                    remove_prefix_suffix(val.group(2).strip(), '"', '"')
                )
            ins = {"type": bib_type, "key": bib_key, "content": bib_content}

            if bib_type == "article":
                ins["content"]["volume"] = ins["content"]["journal"]
            elif bib_type == "inproceedings":
                ins["content"]["volume"] = ins["content"]["booktitle"]

            ins["id"] = ins["key"]
            ins["title"] = ins["content"]["title"]
            ins["abstract"] = ins["content"].get("abstract", "")
            ins["categories"] = ["cs.CL"]
            year = ins["content"].get("year", "2024")
            month = parse_bib_month(ins["content"].get("month", "jan"))
            day = "01"
            ins["update_date"] = f"{year}-{month}-{day}"

            data.append(ins)

    dump_jsonlines(data, output_filepath)


if __name__ == "__main__":
    parse_bib(
        "anthology+abstracts-20241008.bib.gz",
        "anthology+abstracts-20241008.json"
    )
