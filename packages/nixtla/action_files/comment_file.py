import os

import fire
import requests

token = os.environ["GITHUB_TOKEN"]
pr_number = os.environ["PR_NUMBER"]
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json",
}
base_url = "https://api.github.com/repos/Nixtla/nixtla/issues"


def get_comments():
    resp = requests.get(f"{base_url}/{pr_number}/comments", headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(resp.text)
    return resp.json()


def upsert_comment(body: str, comment_id: str | None):
    data = {"body": body}
    if comment_id is None:
        resp = requests.post(
            f"{base_url}/{pr_number}/comments", json=data, headers=headers
        )
    else:
        resp = requests.patch(
            f"{base_url}/comments/{comment_id}", json=data, headers=headers
        )
    return resp


def main(search_term: str, file: str):
    comments = get_comments()
    existing_comment = [
        c for c in comments if search_term in c["body"] and c["user"]["type"] == "Bot"
    ]
    if existing_comment:
        comment_id = existing_comment[0]["id"]
    else:
        comment_id = None
    with open(file, "rt") as f:
        summary = f.read()
    resp = upsert_comment(summary, comment_id)
    if resp.status_code not in (200, 201, 202):
        raise RuntimeError(f"{resp.status_code}: {resp.text}")


if __name__ == "__main__":
    fire.Fire(main)
