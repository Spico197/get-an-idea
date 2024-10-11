import requests
from pathlib import Path
from tqdm import tqdm

from src.utils.io import load_json


def download_pdf(url: str, save_path: str):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192

    file_path = Path(save_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as file, tqdm(
        desc=file_path.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        ncols=75,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)


if __name__ == "__main__":
    papers = load_json("data/llm_safety_list.json")
    out_dir = Path("data/llm_safety_pdfs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for paper in tqdm(papers, desc="Downloading PDFs"):
        url = f"https://arxiv.org/pdf/{paper['id']}.pdf"
        download_pdf(url, out_dir / f"{paper['id']}.pdf")
