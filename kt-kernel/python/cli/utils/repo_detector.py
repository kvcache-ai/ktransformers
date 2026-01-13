"""
Repo Detector

Automatically detect repository information from model README.md files
"""

import re
from pathlib import Path
from typing import Optional, Dict, Tuple
import yaml


def parse_readme_frontmatter(readme_path: Path) -> Optional[Dict]:
    """
    Parse YAML frontmatter from README.md

    Args:
        readme_path: Path to README.md file

    Returns:
        Dictionary of frontmatter data, or None if not found
    """
    if not readme_path.exists():
        return None

    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Match YAML frontmatter between --- markers
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not match:
            return None

        yaml_content = match.group(1)

        # Parse YAML
        try:
            data = yaml.safe_load(yaml_content)
            return data if isinstance(data, dict) else None
        except yaml.YAMLError:
            return None

    except Exception as e:
        return None


def extract_repo_from_frontmatter(frontmatter: Dict) -> Optional[Tuple[str, str]]:
    """
    Extract repo_id and repo_type from frontmatter

    Args:
        frontmatter: Parsed YAML frontmatter dictionary

    Returns:
        Tuple of (repo_id, repo_type) or None
        repo_type is either "huggingface" or "modelscope"
    """
    if not frontmatter:
        return None

    # Priority 1: Extract from license_link (most reliable)
    license_link = frontmatter.get("license_link")
    if license_link and isinstance(license_link, str):
        result = _extract_repo_from_url(license_link)
        if result:
            return result

    # Priority 2: Try to find repo_id from other fields
    repo_id = None

    # Check base_model field
    base_model = frontmatter.get("base_model")
    if base_model:
        if isinstance(base_model, list) and len(base_model) > 0:
            # base_model is a list, take first item
            repo_id = base_model[0]
        elif isinstance(base_model, str):
            repo_id = base_model

    # Check model-index field
    if not repo_id:
        model_index = frontmatter.get("model-index")
        if isinstance(model_index, list) and len(model_index) > 0:
            first_model = model_index[0]
            if isinstance(first_model, dict):
                repo_id = first_model.get("name")

    # Check model_name field
    if not repo_id:
        repo_id = frontmatter.get("model_name")

    if not repo_id or not isinstance(repo_id, str):
        return None

    # Validate format: should be "namespace/model-name"
    if "/" not in repo_id:
        return None

    parts = repo_id.split("/")
    if len(parts) != 2:
        return None

    # Determine repo type
    repo_type = "huggingface"  # Default

    # Look for ModelScope indicators
    if "modelscope" in repo_id.lower():
        repo_type = "modelscope"

    # Check tags
    tags = frontmatter.get("tags", [])
    if isinstance(tags, list):
        if "modelscope" in [str(t).lower() for t in tags]:
            repo_type = "modelscope"

    return (repo_id, repo_type)


def _extract_repo_from_url(url: str) -> Optional[Tuple[str, str]]:
    """
    Extract repo_id and repo_type from a URL

    Supports:
    - https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/LICENSE
    - https://modelscope.cn/models/Qwen/Qwen3-30B-A3B

    Args:
        url: URL string

    Returns:
        Tuple of (repo_id, repo_type) or None
    """
    # HuggingFace pattern: https://huggingface.co/{namespace}/{model}/...
    hf_match = re.match(r"https?://huggingface\.co/([^/]+)/([^/]+)", url)
    if hf_match:
        namespace = hf_match.group(1)
        model_name = hf_match.group(2)
        repo_id = f"{namespace}/{model_name}"
        return (repo_id, "huggingface")

    # ModelScope pattern: https://modelscope.cn/models/{namespace}/{model}
    ms_match = re.match(r"https?://(?:www\.)?modelscope\.cn/models/([^/]+)/([^/]+)", url)
    if ms_match:
        namespace = ms_match.group(1)
        model_name = ms_match.group(2)
        repo_id = f"{namespace}/{model_name}"
        return (repo_id, "modelscope")

    return None


def extract_repo_from_global_search(readme_path: Path) -> Optional[Tuple[str, str]]:
    """
    Extract repo info by globally searching for URLs in README.md

    Args:
        readme_path: Path to README.md file

    Returns:
        Tuple of (repo_id, repo_type) or None if not found
    """
    if not readme_path.exists():
        return None

    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find all HuggingFace URLs
        hf_pattern = r"https?://huggingface\.co/([^/\s]+)/([^/\s\)]+)"
        hf_matches = re.findall(hf_pattern, content)

        # Find all ModelScope URLs
        ms_pattern = r"https?://(?:www\.)?modelscope\.cn/models/([^/\s]+)/([^/\s\)]+)"
        ms_matches = re.findall(ms_pattern, content)

        # Collect all found repos with their types
        found_repos = []

        for namespace, model_name in hf_matches:
            # Skip common non-repo paths
            if namespace.lower() in ["docs", "blog", "spaces", "datasets"]:
                continue
            if model_name.lower() in ["tree", "blob", "raw", "resolve", "discussions"]:
                continue

            repo_id = f"{namespace}/{model_name}"
            found_repos.append((repo_id, "huggingface"))

        for namespace, model_name in ms_matches:
            repo_id = f"{namespace}/{model_name}"
            found_repos.append((repo_id, "modelscope"))

        if not found_repos:
            return None

        # If multiple different repos found, use the last one
        # First, deduplicate
        seen = {}
        for repo_id, repo_type in found_repos:
            seen[repo_id] = repo_type  # Will keep the last occurrence

        # Get the last unique repo
        if seen:
            # Use the last item from found_repos that's unique
            last_unique = None
            for repo_id, repo_type in found_repos:
                if repo_id in seen:
                    last_unique = (repo_id, repo_type)

            return last_unique

        return None

    except Exception as e:
        return None


def detect_repo_for_model(model_path: str) -> Optional[Tuple[str, str]]:
    """
    Detect repository information for a model

    Strategy:
    Only extract from YAML frontmatter metadata in README.md
    (Removed global URL search to avoid false positives)

    Args:
        model_path: Path to model directory

    Returns:
        Tuple of (repo_id, repo_type) or None if not detected
    """
    model_dir = Path(model_path)

    if not model_dir.exists() or not model_dir.is_dir():
        return None

    # Look for README.md
    readme_path = model_dir / "README.md"
    if not readme_path.exists():
        return None

    # Only parse YAML frontmatter (no fallback to global search)
    frontmatter = parse_readme_frontmatter(readme_path)
    if frontmatter:
        return extract_repo_from_frontmatter(frontmatter)

    return None


def scan_models_for_repo(model_list) -> Dict:
    """
    Scan a list of models and detect repo information

    Args:
        model_list: List of UserModel objects

    Returns:
        Dictionary with scan results:
        {
            'detected': [(model, repo_id, repo_type), ...],
            'not_detected': [model, ...],
            'skipped': [model, ...]  # Already has repo_id
        }
    """
    results = {"detected": [], "not_detected": [], "skipped": []}

    for model in model_list:
        # Skip if already has repo_id
        if model.repo_id:
            results["skipped"].append(model)
            continue

        # Only process safetensors and gguf models
        if model.format not in ["safetensors", "gguf"]:
            results["skipped"].append(model)
            continue

        # Try to detect repo
        repo_info = detect_repo_for_model(model.path)

        if repo_info:
            repo_id, repo_type = repo_info
            results["detected"].append((model, repo_id, repo_type))
        else:
            results["not_detected"].append(model)

    return results


def format_detection_report(results: Dict) -> str:
    """
    Format scan results into a readable report

    Args:
        results: Results from scan_models_for_repo()

    Returns:
        Formatted string report
    """
    lines = []

    lines.append("=" * 80)
    lines.append("Auto-Detection Report")
    lines.append("=" * 80)
    lines.append("")

    # Detected
    if results["detected"]:
        lines.append(f"✓ Detected repository information ({len(results['detected'])} models):")
        lines.append("")
        for model, repo_id, repo_type in results["detected"]:
            lines.append(f"  • {model.name}")
            lines.append(f"    Path: {model.path}")
            lines.append(f"    Repo: {repo_id} ({repo_type})")
            lines.append("")

    # Not detected
    if results["not_detected"]:
        lines.append(f"✗ No repository information found ({len(results['not_detected'])} models):")
        lines.append("")
        for model in results["not_detected"]:
            lines.append(f"  • {model.name}")
            lines.append(f"    Path: {model.path}")
        lines.append("")

    # Skipped
    if results["skipped"]:
        lines.append(f"⊘ Skipped ({len(results['skipped'])} models):")
        lines.append(f"  (Already have repo_id or not safetensors/gguf format)")
        lines.append("")

    lines.append("=" * 80)
    lines.append(
        f"Summary: {len(results['detected'])} detected, "
        f"{len(results['not_detected'])} not detected, "
        f"{len(results['skipped'])} skipped"
    )
    lines.append("=" * 80)

    return "\n".join(lines)


def apply_detection_results(results: Dict, registry) -> int:
    """
    Apply detected repo information to models in registry

    Args:
        results: Results from scan_models_for_repo()
        registry: UserModelRegistry instance

    Returns:
        Number of models updated
    """
    updated_count = 0

    for model, repo_id, repo_type in results["detected"]:
        success = registry.update_model(model.name, {"repo_id": repo_id, "repo_type": repo_type})

        if success:
            updated_count += 1

    return updated_count
