import os
import re
import sys
import json
import time
from pathlib import Path
from datetime import datetime

from tqdm import tqdm
from loguru import logger
from zhipuai import ZhipuAI
from dotenv import load_dotenv

from src.utils.io import load_json, dump_jsonlines
from src.utils.downloading import download_pdf

load_dotenv()

API_KEY = os.getenv("ZHIPUAI_API_KEY")
# glm-4-flash is a free model
MODEL_TYPE = "glm-4-flash"
PROMPT = """介绍一下作者单位、前人工作以及他们仍未解决的问题、这篇论文在研究的主要问题、对应的解决方案、新的结论或发现、当前方法存在的局限性和其它可以继续研究的内容，并以JSON的格式返回。注意内容要尽可能回答完全，不要出现省略号。
返回格式的例子如下，注意不要照抄例子里的内容，要结合论文的实际内容进行总结：
```json
{
  "author_affiliations": ["Stanford University", "Meta AI"],
  "related_work": [
    {"task": "命名实体识别", "problem": "数据中存在漏标，导致模型的recall较低", "solution": "使用人工标注方法……", "remaining_problem": "人工方法的耗时极大，导致……"},
    {"task": "命名实体识别", "problem": "数据中存在漏标，导致模型的recall较低", "solution": "使用交叉验证方法……", "remaining_problem": "该方法一定程度上缓解了……，但仍存在……"}
  ],
  "method": [
    {"task": "命名实体识别数据集构建", "problem": "由漏标导致的数据质量问题", "solution": "采用半监督方法对标签进行补齐，具体使用了……"},
    {"task": "命名实体识别方法增强", "problem": "模型语义理解不够的问题", "solution": "采用xxx方法增强……"},
    {"task": "篇章事件抽取", "problem": "事件要素边界抽取错误", "solution": "使用xxx方法……"}
  ],
  "experiment": {
    "dataset": ["dataset1", "dataset2"],
    "metrics": ["Pass@1", "F1"],
    "analyses": ["数据长度对结果的影响", "不同实体类别的具体结果", "方法的上限"]
  },
  "findings_and_conclusions": [
    "大模型相较于小模型的效果提升非常明显",
    "其它发现或结论"
  ],
  "open_questions": [
    {"task": "命名实体识别的xxx", "problem": "尚未解决的问题1"},
    {"task": "命名实体识别的xxx", "problem": "尚未解决的问题2"}
  ]
}
```"""


class ModelAPI(object):
    def __init__(self, model, api_key) -> None:
        self.model = model
        self.api_key = api_key

        self.client = ZhipuAI(api_key=api_key)

    def extract_file_content(self, filepath: str) -> str:
        # 大小：单个文件50M、总数限制为100个文件
        file_object = self.client.files.create(
            file=Path(filepath), purpose="file-extract"
        )
        file_content = self.client.files.content(file_id=file_object.id).content
        file_content = json.loads(file_content)["content"]
        _ = self.client.files.delete(file_id=file_object.id)
        return file_content

    def chat(self, prompt: str):
        logger.debug(f"Model: {self.model}\nPrompt: {prompt}")
        response = self.client.chat.completions.create(
            model=self.model,  # 请填写您要调用的模型名称
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        response_message = response.choices[0].message.content
        logger.debug(f"Response: {response_message}")
        return response_message

    def parse_json(self, string: str) -> dict:
        obj = re.search(r"```json(.*?)```", string, re.DOTALL)
        if obj:
            return json.loads(obj.group(1))
        else:
            return None

    def summarize_paper(self, paper_filepath: str):
        paper_content = self.extract_file_content(paper_filepath)
        summary_prompt = f"{PROMPT}\n论文内容：\n{paper_content}"
        response = self.chat(summary_prompt)
        result = self.parse_json(response)
        return result

    def get_messages(self, paper_filepath: str) -> list:
        paper_content = self.extract_file_content(paper_filepath)
        summary_prompt = f"{PROMPT}\n论文内容：\n{paper_content}"
        msg_list = [
            {"role": "user", "content": summary_prompt},
        ]
        return msg_list

    def get_ins_for_batch(self, paper_filepath: str) -> dict:
        filename = Path(paper_filepath).name
        paper_id = re.sub(r"v(\d+?).pdf", "", filename)
        return {
            "custom_id": paper_id,
            "method": "POST",
            "url": "/v4/chat/completions",
            "body": {
                "model": self.model,  # 每个batch文件只能包含对单个模型的请求,支持 glm-4-0520 glm-4-air、glm-4-flash、glm-4、glm-3-turbo.
                "messages": self.get_messages(paper_filepath),
            },
        }


def get_download_links(input_filepath, output_filepath):
    links = []
    with open(input_filepath, "r", encoding="utf8") as fin:
        data = json.load(fin)
        for ins in data:
            links.append(f"https://arxiv.org/pdf/{ins['id']}.pdf")
    with open(output_filepath, "w", encoding="utf8") as fout:
        for link in links:
            fout.write(f"{link}\n")


def find_difference(parent_file, subset_file):
    import re

    def replace_url(url):
        return re.sub(r"v(\d+?).pdf", ".pdf", url)

    with open(parent_file, "r") as file:
        parent_urls = set([replace_url(url) for url in file.read().splitlines()])

    with open(subset_file, "r") as file:
        subset_urls = set([replace_url(url) for url in file.read().splitlines()])

    difference = parent_urls - subset_urls
    return difference


def summary_main():
    global MODEL_TYPE
    global API_KEY

    model = ModelAPI(MODEL_TYPE, API_KEY)
    filepaths = list(Path("llm_safety_papers").glob("*.pdf"))
    id2path = {}
    for path in filepaths:
        paper_id = re.sub(r"v(\d+?).pdf", "", path.name)
        id2path[paper_id] = path

    with open("llm_safety_list.json", "r", encoding="utf8") as fin:
        papers = json.load(fin)

    results = []
    for paper in tqdm(papers, ncols=80):
        pdf_filepath = id2path.get(paper["id"])
        if pdf_filepath:
            try:
                # summary_result = model.summarize_paper(pdf_filepath)
                # paper["model_summary"] = summary_result
                batch_info = model.get_ins_for_batch(pdf_filepath)
                results.append(batch_info)
            except Exception:
                logger.debug(f"ERR {paper['id']} - {paper['title']}")
        else:
            logger.debug(f"SKIP {paper['id']} - {paper['title']}")
        # results.append(paper)

    # with open("llm_safety_paper_summaries.json", "w", encoding="utf8") as fout:
    #     json.dump(results, fout, indent=2, ensure_ascii=False)

    with open("llm_safety_paper_batch_info.jsonl", "w", encoding="utf8") as fout:
        for ins in results:
            fout.write(f"{json.dumps(ins, ensure_ascii=False)}\n")


def summarize_subsets(subsets, data_info_dir, result_dir):
    output_dir = Path(result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    model = ModelAPI(MODEL_TYPE, API_KEY)

    for subset in subsets:
        data_info = load_json(f"{data_info_dir}/{subset}.json")
        (output_dir / subset).mkdir(parents=True, exist_ok=True)
        mid_info_path = output_dir / subset / "mid_info.json"
        final_info_path = output_dir / subset / "final_result.json"
        mid_info_f = open(mid_info_path, "a", encoding="utf8")
        for paper in tqdm(data_info, ncols=80, desc=subset):
            paper_id = paper["id"]
            paper_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            out_pdf_path = pdf_dir / f"{paper_id}.pdf"
            max_retries = 5
            while max_retries > 0:
                try:
                    download_pdf(paper_url, out_pdf_path, show_progress_bar=False)
                    result = model.summarize_paper(out_pdf_path)
                    paper.update({"summary_results": result})
                    mid_info_f.write(f"{json.dumps(paper, ensure_ascii=False, indent=2)}\n")
                    break
                except Exception as err:
                    logger.error(f"ERR: {err}")
                max_retries -= 1
                time.sleep(10)
        mid_info_f.close()
        dump_jsonlines(data_info, final_info_path)


if __name__ == "__main__":
    # model = ModelAPI(model_type, api_key)
    # paper_path = "2408.12076v1.pdf"
    # import time
    # s = time.time()
    # result = model.summarize_paper(paper_path)
    # tt = time.time() - s
    # print(result)
    # print(tt)  # 15s

    # get_download_links("llm_safety_list.json", "llm_safety_arxiv_url.txt")

    # difference = find_difference('llm_safety_arxiv_url.txt', 'pdflist.txt')
    # for url in difference:
    #     print(url)

    # summary_main()

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    Path("logs").mkdir(parents=True, exist_ok=True)
    current_datetime = datetime.now().strftime("%b %d %Y %H-%M-%S")
    logger.add(f"logs/{current_datetime}.log")

    subsets = ["tool", "task_planning", "docee", "ee", "gui", "moe"]
    data_info_dir = "results/arxiv/1900-11-25"
    result_dir = "results/arxiv/1900-11-25/summaries"
    summarize_subsets(subsets, data_info_dir, result_dir)
