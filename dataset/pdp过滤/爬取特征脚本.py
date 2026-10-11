import requests
import time
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os
import base64

# ===================== 配置区 =====================
USE_PROXY = False
PROXY_HOST = "127.0.0.1"
PROXY_PORT = "7890"

GITHUB_TOKENS = [
    ""
]
INPUT_EXCEL_PATH = r"C:\Users\16549\Desktop\开源软件会议论文\爬取2022年之后并被打上标签的no-ai项目\human_no_ai_java_repos_1000.xlsx"
SHEET_NAME = "human_no_ai_java_repos_1000"
REPO_NAME_COL = "repo_name"
OUTPUT_CSV_PATH = "pdp_no-ai-java_features_106.csv"
SAVE_STEP = 5
THREAD_NUM = 5
MAX_DELAY = 0.4
# ============================================================

session = requests.Session()
if USE_PROXY:
    PROXIES = {
        "http": f"http://{PROXY_HOST}:{PROXY_PORT}",
        "https": f"https://{PROXY_HOST}:{PROXY_PORT}",
    }
    session.proxies.update(PROXIES)
else:
    session.trust_env = False
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
        if key in os.environ:
            del os.environ[key]

# =====================================================================
# 特征定义：与 J48_Level_readme_PDP.model 内部反射出的 106 个属性一一对应
# （属性名来自对 .model 文件的反射解析，class 属性 "pdp" 不在这里生成，
#  是预测时才需要的占位列）
# =====================================================================

# ---- Description keywords（论文 Table 4, Description 关键词表）----
# 关键词 -> 对应模型属性名（多数同名，个别有拼写变体，按模型反射结果对齐）
DESC_KEYWORD_MAP = {
    "mirror": "mirror",
    "fork": "fork",
    "moved": "moved",
    "longer": "longer",          # "no longer" 里的 longer
    "test": "test",
    "personal": "personal",
    "website": "website",
    "framework": "framework",
    "tool": "tool",
    "module": "module",
    "component": "component",
    "app": "app",
    "system": "system",
    "dotfiles": "dotfiles",
    "collection": "collection",
    "blog": "blog",
    "plugin": "plugin",
    "library": "library",
    "server": "server",
    "config": "config",
    "guide": "guide",
    "set": "set",
    "repository": "repository",
    "deprecated": "deprecated",
    "file": "file",
    "demo": "demo",
    "my": "my",
    "github": "github",
    "dot": "dot",
    "simple": "simple",
    "extension": "extension",
    "helper": "helper",
    "template": "template",
    "http": "http",
    "https": "https",
    "source": "source",
    "setting": "setting",
    "list of": "listof",
    "collection of": "collectionof",
    "set of": "setof",
    "example": "example",
    "vim": "vim",
    "sample": "sample",
    "university": "university",
    "school": "school",
    "practice": "practice",
    "backup": "backup",
    "intro": "intro",
    "first": "first",
    "tutorial": "Tutorial",
    "course": "course",
    "copy": "copy",
    "null": "nodescription",     # 论文里 "null" 关键词，模型属性名叫 nodescription
    "localisation": "localization",
    "storage": "storage",
    "theme": "theme",
    "resume": "resume",
    "clone": "clone",
    "translation": "translation",
    "obsolete": "obsole",
    "unmaintained": "unmaintained",
    "inactive": "inactive",
    "cookbook": "cookbook",
    "retired": "retired",
    "exported from": "exportedfrom",
    "abandoned": "abondoned",
    "documentation": "documentation",  # "abandoned documentation" 短语里的 documentation
}

# ---- URL keywords（论文 Table 4, URL 行）----
URL_KEYWORD_MAP = {
    "dot": "url_dot",       # url 中出现 "." 相关配置词，按论文原文是 dot/config/doc 三个 URL 关键词
    "config": "url_config",
    "doc": "url_doc",
}

# ---- README section / obvious keywords（论文 Table 5）----
# 值为 (匹配用的关键词或短语列表, 模型属性名)
README_KEYWORD_MAP = {
    "readme_wiki": ["wiki"],
    "readme_website": ["website"],
    "readme_setup": ["setup"],
    "readme_install": ["install"],
    "readme_usage": ["usage"],
    "readme_howtouse": ["how to use"],
    "readme_howto": ["how to", "build"],   # howto 泛化匹配 "how to"
    "readme_build": ["build"],
    "readme_contributing": ["contributing"],
    "readme_contact": ["contact"],
    "readme_author": ["author"],
    "readme_reportbug": ["report bug"],
    "readme_www": ["www"],
    "readme_http": ["http"],
    "readme_obsole": ["obsolete"],
    "readme_unmaintained": ["unmaintained"],
    "readme_inactive": ["inactive"],
    "readme_nolonger": ["no longer"],
    "readme_deprecated": ["deprecated"],
    "readme_abondoned": ["abandoned"],
    "readme_cookbook": ["cookbook"],
    "readme_retired": ["retired"],
    "readme_mirror": ["mirror"],
    "readme_sample": ["sample"],
    "readme_demo": ["demo"],
}

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

save_lock = Lock()
finished_repos = set()
result_rows = []


def load_finished():
    global finished_repos, result_rows
    try:
        df = pd.read_csv(OUTPUT_CSV_PATH)
        result_rows = df.to_dict("records")
        finished_repos = set(row["repo_name"] for row in result_rows)
        print(f"断点加载完成，已爬取 {len(finished_repos)} 个仓库")
    except FileNotFoundError:
        print("无历史文件，全新爬取")


def save_result():
    df = pd.DataFrame(result_rows)
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")


def check_and_save():
    with save_lock:
        if len(result_rows) > 0 and len(result_rows) % SAVE_STEP == 0:
            save_result()
            print(f"已保存断点，累计完成 {len(result_rows)}")


class TokenPool:
    def __init__(self, token_list):
        self.tokens = token_list
        self.token_idx = 0
        self.token_lock = Lock()

    def get_available_token(self):
        while True:
            with self.token_lock:
                current_token = self.tokens[self.token_idx]
                self.token_idx = (self.token_idx + 1) % len(self.tokens)
            headers = {"Authorization": f"token {current_token}", "Accept": "application/vnd.github.v3+json"}
            try:
                resp = session.get("https://api.github.com/rate_limit", headers=headers, timeout=30)
                if resp.status_code != 200:
                    time.sleep(2)
                    continue
                limit_data = resp.json()
                remaining = limit_data["rate"]["remaining"]
                reset_ts = limit_data["rate"]["reset"]
                if remaining > 50:
                    return current_token
                now = time.time()
                sleep_time = int(reset_ts - now) + 10
                print(f"Token余量耗尽，休眠 {sleep_time}s 后重试")
                time.sleep(sleep_time)
            except Exception as e:
                print(f"获取 rate_limit 失败: {e}，等待5秒后重试")
                time.sleep(5)


token_pool = TokenPool(GITHUB_TOKENS)


def get_headers():
    token = token_pool.get_available_token()
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }


def get_repo_basic(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        headers = get_headers()
        resp = session.get(url, headers=headers, timeout=30)
        time.sleep(MAX_DELAY)
        if resp.status_code == 404:
            return None, "not_found"
        resp.raise_for_status()
        return resp.json(), "ok"
    except Exception as e:
        print(f"仓库基础接口异常 {owner}/{repo}: {str(e)}")
        return None, "error"


def get_pr_total(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all&per_page=1"
    try:
        headers = get_headers()
        resp = session.get(url, headers=headers, timeout=30)
        time.sleep(MAX_DELAY)
        if "Link" not in resp.headers:
            return len(resp.json())
        match = re.search(r'page=(\d+)>; rel="last"', resp.headers["Link"])
        return int(match.group(1)) if match else len(resp.json())
    except Exception:
        return 0


def get_committer_count(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors?per_page=1"
    try:
        headers = get_headers()
        resp = session.get(url, headers=headers, timeout=30)
        time.sleep(MAX_DELAY)
        if "Link" not in resp.headers:
            return len(resp.json())
        match = re.search(r'page=(\d+)>; rel="last"', resp.headers["Link"])
        cnt = int(match.group(1)) if match else len(resp.json())
        return min(cnt, 500)
    except Exception:
        return 0


def get_readme_content(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
        headers = get_headers()
        resp = session.get(url, headers=headers, timeout=30)
        time.sleep(MAX_DELAY)

        if resp.status_code == 404:
            return ""
        if resp.status_code != 200:
            print(f"README接口异常 {owner}/{repo}: status={resp.status_code} body={resp.text[:200]}")
            return ""

        data = resp.json()
        content_b64 = data.get("content")
        if not content_b64:
            print(f"README无content字段 {owner}/{repo}: keys={list(data.keys())}")
            return ""

        try:
            raw_bytes = base64.b64decode(content_b64.replace("\n", ""))
            return raw_bytes.decode("utf-8", errors="ignore")
        except Exception as decode_err:
            print(f"README base64解码失败 {owner}/{repo}: {decode_err}")
            return ""

    except Exception as e:
        print(f"README获取异常 {owner}/{repo}: {type(e).__name__}: {e}")
        return ""


# =====================================================================
# 特征计算函数（106 列细粒度版本）
# =====================================================================

def parse_desc_features(desc_text: str) -> dict:
    """
    对应论文 Table 4 Description 关键词表，逐词一列（0/1）。
    使用 \\b 词边界匹配，避免子串误匹配（如 'test' 不会命中 'testing123' 之外的东西，
    但 'set' 这种短词容易在长文本里误命中，这是论文方法本身的局限，这里如实复现）。
    """
    text = (desc_text or "").lower()
    feat = {}
    for keyword, attr_name in DESC_KEYWORD_MAP.items():
        if " " in keyword:
            hit = 1 if keyword in text else 0
        else:
            # 论文原文是用 MySQL 的 '%keyword%' 子串匹配（Section 4.2.1），
            # 即不要求完整单词边界，"frameworks" 也会命中 "framework"。
            # 这里改用子串匹配以忠实复现论文口径（而不是更严格的 \b 词边界）。
            hit = 1 if keyword in text else 0
        # 同一个属性名可能被多个关键词写入（如 documentation 只在一个短语里出现），
        # 用 or 合并，保证已经写过 1 的不会被后续 0 覆盖
        feat[attr_name] = max(feat.get(attr_name, 0), hit)
    return feat


def parse_url_features(url_text: str) -> dict:
    """对应论文 Table 4 URL 行：dot / config / doc 三个关键词"""
    text = (url_text or "").lower()
    feat = {}
    for keyword, attr_name in URL_KEYWORD_MAP.items():
        feat[attr_name] = 1 if keyword in text else 0
    return feat


def parse_readme_features(readme_text: str) -> dict:
    """
    对应论文 Table 5：
      - word: README 字数
      - have_at: 是否包含 @ （联系邮箱线索）
      - 逐 section / obvious 关键词 0/1（前缀 readme_）
    """
    text = readme_text or ""
    text_low = text.lower()

    feat = {
        "word": len(text.split()) if text else 0,
        "have_at": 1 if "@" in text else 0,
    }

    for attr_name, keywords in README_KEYWORD_MAP.items():
        hit = 0
        for kw in keywords:
            # 同样按论文 Section 4.2.1 的 '%keyword%' 子串匹配口径处理
            if kw in text_low:
                hit = 1
                break
        feat[attr_name] = hit

    return feat


def build_feature_row(repo_name):
    global result_rows, finished_repos
    if repo_name in finished_repos:
        return None
    try:
        owner, repo = repo_name.split("/")
    except ValueError:
        print(f"格式错误跳过: {repo_name}")
        row = {"repo_name": repo_name}
        with save_lock:
            result_rows.append(row)
        return row

    repo_data, status = get_repo_basic(owner, repo)
    if status != "ok" or (repo_data and repo_data.get("private") is True):
        row = {"repo_name": repo_name}
        with save_lock:
            result_rows.append(row)
        return row

    # ---- Basic features（论文 Table 3）----
    star = repo_data.get("stargazers_count", 0) or 0
    watcher = repo_data.get("subscribers_count", 0) or 0
    committer = get_committer_count(owner, repo)
    comunity = committer  # 与原脚本一致：community 近似用 committer 数（GHTorrent 原定义更复杂，这里保持你原脚本的简化口径）
    pull_request = get_pr_total(owner, repo)
    have_language = 1 if repo_data.get("language") else 0

    basic_feats = {
        "repo_name": repo_name,
        "committer": committer,
        "comunity": comunity,
        "star": star,
        "watcher": watcher,
        "pull_request": pull_request,
        "have_language": have_language,
    }

    # ---- Description + URL keywords（Table 4）----
    desc = repo_data.get("description", "") or ""
    desc_feats = parse_desc_features(desc)

    html_url = repo_data.get("html_url", "") or ""
    homepage = repo_data.get("homepage", "") or ""
    url_text = f"{html_url} {homepage}"
    url_feats = parse_url_features(url_text)

    # ---- README features（Table 5）----
    readme_txt = get_readme_content(owner, repo)
    readme_feats = parse_readme_features(readme_txt)

    # is_null: description 为空 -> 1，否则 0（论文里 'null' 关键词本身已在 desc_feats
    # 的 nodescription 属性里处理字面 "null" 一词命中；is_null 这里额外表示"确实没有描述"）
    is_null_feat = {"is_null": 1 if not desc.strip() else 0}

    full_row = {}
    full_row.update(basic_feats)
    full_row.update(desc_feats)
    full_row.update(url_feats)
    full_row.update(readme_feats)
    full_row.update(is_null_feat)

    with save_lock:
        result_rows.append(full_row)
        finished_repos.add(repo_name)

    check_and_save()
    return full_row


def main():
    load_finished()
    df_input = pd.read_excel(INPUT_EXCEL_PATH, sheet_name=SHEET_NAME)
    all_repos = df_input[REPO_NAME_COL].dropna().unique().tolist()
    todo_list = [r for r in all_repos if r not in finished_repos]
    print(f"总仓库: {len(all_repos)} | 剩余待爬: {len(todo_list)} | 并发线程: {THREAD_NUM}")

    with ThreadPoolExecutor(max_workers=THREAD_NUM) as executor:
        futures = [executor.submit(build_feature_row, repo) for repo in todo_list]
        for idx, future in enumerate(as_completed(futures)):
            res = future.result()
            if res:
                print(f"进度 {idx + 1}/{len(todo_list)} | 完成: {res['repo_name']}")

    with save_lock:
        save_result()
    print(f"\n全部爬取结束！输出文件: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()