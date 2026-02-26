"""
OAS REST Standards Reviewer
----------------------------
Fetches changed OAS files from a GitHub PR, sends them to GPT-4 for review
against REST standards, and posts inline PR comments for each violation.
"""

import os
import json
import subprocess
import sys
import yaml
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PR_NUMBER = os.environ["PR_NUMBER"]
REPO = os.environ["REPO"]          # e.g. "org/repo"
BASE_SHA = os.environ["BASE_SHA"]
HEAD_SHA = os.environ["HEAD_SHA"]

GH_API = "https://api.github.com"
GH_HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

OAS_EXTENSIONS = {".yaml", ".yml", ".json"}

client = OpenAI(api_key=OPENAI_API_KEY)

# ── REST Standards (customise these to match your org's ruleset) ──────────────
REST_STANDARDS = """
You are an expert API designer performing an automated code review of OpenAPI Specification (OAS) files.
Review the provided OAS content and check for violations of the following REST standards:

## Resource Naming
1. Resource names in paths MUST be nouns (e.g. /users, /orders). Verbs are NOT allowed (e.g. /getUser, /createOrder).
2. Resource names MUST be lowercase and use hyphens for multi-word names (e.g. /order-items, NOT /orderItems or /order_items).
3. Resource names MUST be plural (e.g. /users NOT /user), except for singleton resources.
4. Path parameters MUST use camelCase (e.g. {userId}, NOT {user_id} or {UserId}).

## HTTP Methods
5. GET MUST only be used to retrieve/read resources. It MUST NOT modify state.
6. POST MUST be used to create new resources or trigger actions.
7. PUT MUST be used for full resource replacement.
8. PATCH MUST be used for partial resource updates.
9. DELETE MUST be used for resource deletion.

## HTTP Status Codes
10. GET success MUST return 200.
11. POST (create) success MUST return 201 with a Location header.
12. DELETE success MUST return 204 (no body).
13. Validation errors MUST return 400.
14. Authentication errors MUST return 401.
15. Authorization errors MUST return 403.
16. Not found MUST return 404.

## Request / Response Design
17. Request and response bodies MUST be JSON (application/json).
18. Response bodies MUST be objects (not bare arrays at the top level). Use a wrapper key like "data" or "items".
19. Field names in request/response schemas MUST use camelCase.
20. Pagination MUST use query parameters: page, pageSize (or limit/offset).

## General OAS Quality
21. Every endpoint MUST have a summary and description.
22. Every endpoint MUST declare at least one error response (4xx or 5xx).
23. Reusable schemas MUST be defined under components/schemas, not inline.
24. Security schemes MUST be defined and applied globally or per-endpoint.

For each violation found, return a JSON array. Each item MUST have:
- "path": the OAS path key (e.g. "/users/{userId}")
- "method": HTTP method in uppercase, or "general" if not method-specific
- "rule": short rule name
- "severity": "error" | "warning"
- "message": clear explanation of the violation and how to fix it

If there are no violations, return an empty array [].

IMPORTANT: Return ONLY the raw JSON array. No markdown, no explanation outside the JSON.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_changed_oas_files():
    """Return list of (filepath, patch) for OAS files changed in the PR."""
    url = f"{GH_API}/repos/{REPO}/pulls/{PR_NUMBER}/files"
    resp = requests.get(url, headers=GH_HEADERS)
    resp.raise_for_status()
    files = resp.json()

    oas_files = []
    for f in files:
        filename = f["filename"]
        status = f["status"]   # added, modified, removed, renamed
        if status == "removed":
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in OAS_EXTENSIONS:
            continue
        patch = f.get("patch", "")
        oas_files.append({"filename": filename, "patch": patch, "status": status})

    return oas_files


def get_file_content(filepath):
    """Read file content from the checked-out workspace."""
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return None


def is_openapi_file(content: str, filepath: str) -> bool:
    """Check if the file looks like an OpenAPI spec."""
    try:
        if filepath.endswith(".json"):
            data = json.loads(content)
        else:
            data = yaml.safe_load(content)
        return isinstance(data, dict) and ("openapi" in data or "swagger" in data)
    except Exception:
        return False


def review_oas_with_llm(filepath: str, content: str) -> list:
    """Send OAS content to GPT-4 and return list of violation dicts."""
    prompt = f"""
Below is the OpenAPI Specification file: `{filepath}`

```
{content[:12000]}  
```

Review it against the REST standards and return violations as a JSON array.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": REST_STANDARDS},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        violations = json.loads(raw)
        if not isinstance(violations, list):
            return []
        # Filter out any non-dict items GPT may have accidentally included
        return [v for v in violations if isinstance(v, dict)]
    except Exception as e:
        print(f"  ⚠ LLM call failed for {filepath}: {e}")
        return []


def post_pr_comment(violations_by_file: dict):
    """
    Post a single PR issue comment summarising all violations.
    Uses the simple issue comments API — no commit SHA or line numbers needed.
    """
    summary_lines = ["## 🤖 OAS REST Standards Review\n"]

    total_errors = 0
    total_warnings = 0

    for filepath, violations in violations_by_file.items():
        if filepath.startswith("__patch__"):
            continue

        if not violations:
            summary_lines.append(f"\n✅ **`{filepath}`** — No violations found.")
            continue

        errors   = [v for v in violations if v.get("severity") == "error"]
        warnings = [v for v in violations if v.get("severity") == "warning"]
        total_errors   += len(errors)
        total_warnings += len(warnings)

        icon = "❌" if errors else "⚠️"
        summary_lines.append(f"\n{icon} **`{filepath}`** — {len(errors)} error(s), {len(warnings)} warning(s)\n")

        # Group by path for readability
        by_path = {}
        for v in violations:
            by_path.setdefault(v.get("path", "general"), []).append(v)

        for oas_path, path_violations in by_path.items():
            summary_lines.append(f"<details><summary><code>{oas_path}</code></summary>\n")
            for v in path_violations:
                sev_icon = "🔴" if v.get("severity") == "error" else "🟡"
                summary_lines.append(
                    f"\n{sev_icon} **[{v.get('severity','issue').upper()}]** `{v.get('method','general').upper()}` — "
                    f"**{v.get('rule', 'REST Violation')}**\n"
                    f"> {v.get('message', '')}\n"
                )
            summary_lines.append("</details>\n")

    # Overall verdict
    if total_errors == 0 and total_warnings == 0:
        verdict = "✅ **All checks passed!** No REST standard violations found."
    elif total_errors == 0:
        verdict = f"⚠️ **{total_warnings} warning(s) found.** Please review before merging."
    else:
        verdict = f"❌ **{total_errors} error(s) and {total_warnings} warning(s) found.** Please fix before merging."

    summary_lines.insert(1, f"\n{verdict}\n")
    summary_lines.append("\n---\n_Reviewed by OAS REST Standards Bot powered by GPT-4o_")

    body = "\n".join(summary_lines)

    url = f"{GH_API}/repos/{REPO}/issues/{PR_NUMBER}/comments"
    resp = requests.post(url, headers=GH_HEADERS, json={"body": body})

    if resp.status_code in (200, 201):
        print(f"✅ Comment posted ({total_errors} errors, {total_warnings} warnings)")
    else:
        print(f"⚠ Failed to post comment: {resp.status_code} {resp.text}")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🔍 Fetching changed files from PR...")
    changed_files = get_changed_oas_files()

    if not changed_files:
        print("No OAS files changed. Skipping review.")
        sys.exit(0)

    violations_by_file = {}

    for file_info in changed_files:
        filepath = file_info["filename"]
        print(f"\n📄 Reviewing: {filepath}")

        content = get_file_content(filepath)
        if not content:
            print(f"  ⚠ Could not read file (may have been deleted). Skipping.")
            continue

        if not is_openapi_file(content, filepath):
            print(f"  ℹ Not an OpenAPI file. Skipping.")
            continue

        violations = review_oas_with_llm(filepath, content)
        violations_by_file[filepath] = violations
        # Store patch for line-mapping
        violations_by_file[f"__patch__{filepath}"] = file_info["patch"]

        print(f"  Found {len(violations)} violation(s).")
        for v in violations:
            print(f"    [{v.get('severity','?').upper()}] {v.get('rule')} — {v.get('path')} {v.get('method')}")

    # Filter out patch keys for review
    review_data = {k: v for k, v in violations_by_file.items() if not k.startswith("__patch__")}

    if not review_data:
        print("\nNo OAS specs found in the changed files.")
        sys.exit(0)

    print("\n💬 Posting comment to GitHub PR...")
    post_pr_comment(violations_by_file)


if __name__ == "__main__":
    main()
