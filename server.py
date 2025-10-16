import asyncio
import json
import os
from typing import Dict, Tuple, Any
from urllib.parse import urlparse
import httpx

import yaml
from fastmcp import FastMCP


def load_config(path: str = "mcp.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RefResolutionError(Exception):
    pass


def _json_pointer_get(doc: Any, pointer: str) -> Any:
    """Resolve a JSON Pointer like "/a/b/0" against a dict/list doc.
    Empty pointer or '/' points to the root.
    """
    if pointer in ("", "/"):
        return doc
    if not pointer.startswith("/"):
        raise RefResolutionError(f"Invalid JSON Pointer: {pointer}")
    parts = pointer.lstrip("/").split("/")
    cur = doc
    for raw in parts:
        # Unescape per RFC6901
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            try:
                idx = int(key)
            except ValueError:
                raise RefResolutionError(
                    f"Pointer into list requires integer index: {pointer}")
            try:
                cur = cur[idx]
            except IndexError:
                raise RefResolutionError(
                    f"List index out of range in pointer: {pointer}")
        elif isinstance(cur, dict):
            if key not in cur:
                raise RefResolutionError(
                    f"Key '{key}' not found while resolving {pointer}")
            cur = cur[key]
        else:
            raise RefResolutionError(f"Cannot traverse pointer {
                                     pointer} at scalar {cur!r}")
    return cur


def _load_yaml_or_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # try JSON first if it looks like JSON; else YAML
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def resolve_refs(obj: Any, base_dir: str, file_cache: Dict[str, Any] | None = None, seen: set | None = None) -> Any:
    """Recursively resolve $ref across multiple files relative to base_dir.
    - file_cache memoizes parsed external files by absolute path
    - seen protects against cycles for dict nodes with $ref
    Returns a deep-resolved object (no $ref remain).
    """
    if file_cache is None:
        file_cache = {}
    if seen is None:
        seen = set()

    def _resolve_node(node: Any) -> Any:
        # Recurse into lists
        if isinstance(node, list):
            return [_resolve_node(x) for x in node]
        # Dict: check $ref then recurse keys
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                ref = node["$ref"]
                # Cycle guard: use id(node)
                if id(node) in seen:
                    return node
                seen.add(id(node))

                # Split file part and fragment
                if ref.startswith("#"):
                    # Local anchor
                    target = _json_pointer_get(root_doc, ref[1:])
                    resolved = _resolve_node(target)
                else:
                    # External path, possibly with #fragment
                    path_part, frag = (ref.split("#", 1) + [""])[:2]
                    ext_path = os.path.normpath(
                        os.path.join(base_dir, path_part))
                    if ext_path not in file_cache:
                        file_cache[ext_path] = _load_yaml_or_json(ext_path)
                    ext_doc = file_cache[ext_path]
                    target = _json_pointer_get(
                        ext_doc, f"/{frag}" if frag and not frag.startswith("/") else frag)
                    resolved = _resolve_node(target)

                # Merge: resolved replaces the $ref node entirely, but if the node had siblings, they override
                if isinstance(resolved, dict):
                    merged = dict(resolved)
                    for k, v in node.items():
                        if k == "$ref":
                            continue
                        merged[k] = _resolve_node(v)
                    return merged
                else:
                    # Scalar/array replacement; siblings (other keys) are ignored per spec
                    return resolved

            # No $ref: descend into children
            return {k: _resolve_node(v) for k, v in node.items()}
        # Scalars
        return node

    # We need the root_doc for local #/ pointers
    root_doc = obj
    return _resolve_node(obj)


def filter_openapi(spec: Dict, allow: list, deny: list) -> Dict:
    allow_set = set(a.strip().upper() for a in (allow or []))
    deny_set = set(d.strip().upper() for d in (deny or []))

    def mk_key(method: str, path: str) -> str:
        return f"{method.upper()} {path}"

    paths = spec.get("paths", {})
    new_paths = {}
    for path, methods in paths.items():
        kept_methods = {}
        for method, operation in (methods or {}).items():
            if method.lower() not in {"get", "post", "put", "patch", "delete", "head", "options"}:
                continue
            key = mk_key(method, path)
            allowed = (not allow_set) or (key in allow_set)
            denied = key in deny_set
            if allowed and not denied:
                kept_methods[method] = operation
        if kept_methods:
            new_paths[path] = kept_methods
    spec = dict(spec)
    spec["paths"] = new_paths
    return spec

# --------- server factory ---------


async def make_server_from_openapi(cfg: dict) -> FastMCP:
    spec_path = cfg["openapi"]["spec_path"]
    allow = (cfg.get("openapi", {}) or {}).get("allow", [])
    deny = (cfg.get("openapi", {}) or {}).get("deny", [])

    base_dir = os.path.dirname(os.path.abspath(spec_path))

    # Load root spec (YAML/JSON), then resolve $ref across files
    with open(spec_path, "r", encoding="utf-8") as f:
        if spec_path.endswith(".json"):
            raw = json.load(f)
        else:
            raw = yaml.safe_load(f)

    resolved = resolve_refs(raw, base_dir=base_dir)
    filtered = filter_openapi(resolved, allow, deny)

    # Build MCP from resolved, filtered OpenAPI
    resolved = resolve_refs(raw, base_dir=base_dir)
    filtered = filter_openapi(resolved, allow, deny)

    # ---- Build HTTP client for the target API (from servers[0].url or env fallback)
    servers = (filtered.get("servers") or [])
    base_url = None
    if servers and isinstance(servers, list) and isinstance(servers[0], dict):
        base_url = servers[0].get("url")

    if not base_url:
        base_url = (
            cfg.get("openapi").get("base_url")
            or os.environ.get("OPENAPI_BASE_URL")
            or "http://localhost:3000"
        )

    client = httpx.AsyncClient(base_url=base_url, verify=False)

    mcp = FastMCP.from_openapi(
        openapi_spec=filtered,
        client=client,
        name="OpenAPI-MCP",
    )

    @mcp.resource("health://status")
    def health():
        return {"ok": True}

    return mcp


async def run_stdio(mcp: FastMCP):
    mcp.run(transport="stdio")


def run_sse(mcp: FastMCP, host: str, port: int):
    mcp.run(transport="sse", host=host, port=port)  # /sse


def run_streamable_http(mcp: FastMCP, host: str, port: int):
    mcp.run(transport="http", host=host, port=port)


async def main():
    cfg = load_config()
    mcp = await make_server_from_openapi(cfg)

    transports = cfg.get("transports", [])

    # If stdio is present, run it exclusively
    for t in transports:
        if t.get("type") == "stdio":
            await run_stdio(mcp)
            return

    # Otherwise run network transports concurrently
    tasks = []
    for t in transports:
        ttype = t.get("type")
        host = t.get("host", "127.0.0.1")
        port = int(t.get("port", 8000))
        if ttype == "sse":
            tasks.append(asyncio.to_thread(run_sse, mcp, host, port))
        elif ttype in ("streamable-http", "http", "streamable_http"):
            tasks.append(asyncio.to_thread(
                run_streamable_http, mcp, host, port))

    if not tasks:
        raise SystemExit("No transports configured. Check mcp.yaml")

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
        pass
