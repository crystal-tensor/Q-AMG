"""Configure the requested repository's Pages/About using existing Git credentials.

Credentials stay in memory and curl stdin; never printed, written to a file or
embedded in command-line arguments. Default mode only reads repository state.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

REPOSITORY = "crystal-tensor/Q-AMG"
WEBSITE = "https://crystal-tensor.github.io/Q-AMG/"


def credential():
    result = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                            text=True, capture_output=True, check=False,
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if result.returncode != 0 or not token or "\n" in token:
        raise RuntimeError("GitHub Git credentials are not available")
    return token


def request(method, path, payload=None):
    if not path.startswith(f"/repos/{REPOSITORY}"):
        raise ValueError("This deployment helper is scoped to the requested repository")
    token = credential()
    config = "header = "+json.dumps("Authorization: Bearer "+token)+"\n"
    command = ["curl", "--config", "-", "--silent", "--show-error", "--connect-timeout", "20", "--max-time", "60",
               "--request", method, "--header", "Accept: application/vnd.github+json",
               "--header", "X-GitHub-Api-Version: 2026-03-10", "--write-out", "\n%{http_code}"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as body:
        if payload is not None:
            json.dump(payload, body)
            body.flush()
            command += ["--header", "Content-Type: application/json", "--data-binary", "@"+body.name]
        result = subprocess.run([*command, "https://api.github.com"+path], input=config,
                                text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("GitHub API transport failed; credentials were not logged")
    content, _, status = result.stdout.rpartition("\n")
    return int(status), json.loads(content) if content.strip() else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configure-pages", action="store_true")
    parser.add_argument("--set-about", action="store_true")
    args = parser.parse_args()
    status, repo = request("GET", f"/repos/{REPOSITORY}")
    if status != 200:
        raise RuntimeError(f"Repository access failed: HTTP {status}")
    if args.configure_pages:
        status, _ = request("GET", f"/repos/{REPOSITORY}/pages")
        method = "PATCH" if status == 200 else "POST" if status == 404 else None
        if method is None:
            raise RuntimeError(f"Unable to inspect Pages state: HTTP {status}")
        status, result = request(method, f"/repos/{REPOSITORY}/pages", {
            "build_type": "workflow", "source": {"branch": "main", "path": "/"}})
        if status not in {200, 201, 204}:
            raise RuntimeError(f"Pages configuration failed: HTTP {status}; {result.get('message', '')}")
        print(json.dumps({"pages_configured": True, "http_status": status, "url": result.get("html_url")}))
    if args.set_about:
        status, result = request("PATCH", f"/repos/{REPOSITORY}", {
            "homepage": WEBSITE,
            "description": "Quantum-assisted Mechanical Generation: image-to-3D, QuantumBridge structure sampling, disassembly and assembly"})
        if status != 200:
            raise RuntimeError(f"About update failed: HTTP {status}; {result.get('message', '')}")
        print(json.dumps({"about_updated": True, "homepage": result.get("homepage")}))
    if not args.configure_pages and not args.set_about:
        status, pages = request("GET", f"/repos/{REPOSITORY}/pages")
        print(json.dumps({"repository": repo["full_name"], "homepage": repo.get("homepage"),
                          "permissions": repo.get("permissions"), "pages_status": status,
                          "pages_url": pages.get("html_url"), "pages_build_type": pages.get("build_type")}))


if __name__ == "__main__":
    main()
