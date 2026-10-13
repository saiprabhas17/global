"""Validate deployment policy without containers or reading runtime secret files.

Requires Docker Compose with --no-env-resolution and PyYAML. Compose interpolation
and service env-file resolution are both disabled; an explicit temporary non-secret
env file prevents the default .env lookup. Normalized configuration is never printed.
This is a static policy check, not a replacement for nginx -t or Render validation.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def compose_model(files: list[str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gmee-static-config-") as directory:
        env_path = Path(directory) / "nonsecret.env"
        env_path.write_text("GMEE_STATIC_VALIDATION=1\n", encoding="utf-8")
        command = ["docker", "compose", "--env-file", str(env_path)]
        for file in files:
            command.extend(["-f", str(ROOT / file)])
        command.extend(
            [
                "config",
                "--no-env-resolution",
                "--no-interpolate",
                "--format",
                "json",
            ]
        )
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            raise ValidationError(
                "Static Compose validation exceeded 30 seconds."
            ) from None
        except (OSError, subprocess.CalledProcessError):
            # Never echo a resolved model or process environment into CI output.
            raise ValidationError(
                "Compose configuration failed for "
                + ", ".join(files)
                + "; require Compose supporting --no-env-resolution."
            ) from None
    try:
        model = json.loads(result.stdout)
    except ValueError:
        raise ValidationError("Compose did not return a JSON model.") from None
    require(isinstance(model, dict), "Compose model is not an object.")
    print("PASS static Compose: " + ", ".join(files))
    return dict(model)


def validate_proxy_network(model: dict[str, Any]) -> None:
    services = model["services"]
    frontend_ip = services["frontend"]["networks"]["proxy"]["ipv4_address"]
    trusted = services["backend"]["environment"]["FORWARDED_ALLOW_IPS"]
    require(
        trusted == frontend_ip and trusted != "*",
        "Backend must trust only its explicitly addressed nginx proxy peer.",
    )
    require(
        model["networks"]["data"].get("internal") is True,
        "Datastore network must be internal.",
    )
    require(
        "data" not in services["frontend"]["networks"],
        "Frontend must not join the datastore network.",
    )


def validate_compose() -> None:
    production = compose_model(["infra/docker-compose.yml"])
    development = compose_model(
        [
            "infra/docker-compose.yml",
            "infra/docker-compose.dev.yml",
        ]
    )
    tests = compose_model(["infra/docker-compose.test.yml"])
    smoke = compose_model(["infra/docker-compose.smoke.yml"])
    validate_proxy_network(production)
    validate_proxy_network(smoke)
    for store in ("postgres", "neo4j", "redis"):
        require(
            not production["services"][store].get("ports"),
            f"Production {store} must not publish host ports.",
        )
        require(
            set(production["services"][store]["networks"]) == {"data"},
            f"Production {store} must remain on the datastore-only network.",
        )
    backend = production["services"]["backend"]
    require(
        not backend.get("ports"), "Production backend must not publish a host port."
    )
    require(
        all(v.get("type") != "bind" for v in backend.get("volumes", [])),
        "Production backend must not bind-mount host source.",
    )
    require(
        backend["build"].get("target") == "production",
        "Production backend must use its runtime target.",
    )
    env_files = backend.get("env_file", [])
    require(
        any(
            v.get("required") is True and v["path"].endswith(".env") for v in env_files
        ),
        "Runtime root .env must be required, not optional.",
    )
    for service in development["services"].values():
        for port in service.get("ports", []):
            require(
                port.get("host_ip") in {"127.0.0.1", "::1"},
                "Every development host port must bind to loopback.",
            )
    for name in ("backend-tests", "frontend-tests"):
        require(
            tests["services"][name]["build"].get("target") == "test",
            f"{name} must use a dedicated test image target.",
        )
    for name, service in smoke["services"].items():
        require(
            not service.get("ports")
            and not service.get("container_name")
            and not service.get("env_file"),
            f"Smoke {name} must not use host ports, fixed names or runtime env files.",
        )
    for store in ("postgres", "neo4j", "redis"):
        require(
            bool(smoke["services"][store].get("tmpfs"))
            and not smoke["services"][store].get("volumes"),
            f"Smoke {store} must use disposable tmpfs, not persistent database volumes.",
        )
    smoke_service = smoke["services"]["smoke"]
    require(
        smoke_service["environment"].get("GMEE_SMOKE_FIXTURE") == "1"
        and "--fixture-only" in smoke_service["command"],
        "HTTP smoke must require explicit fixture acknowledgement.",
    )
    command = smoke_service["command"]
    require(
        "--deadline-seconds" in command
        and 30 <= int(command[command.index("--deadline-seconds") + 1]) <= 300,
        "HTTP smoke must have a bounded hard deadline.",
    )
    print("PASS Compose isolation, runtime env requirement and proxy trust policy")


def validate_images_and_proxy() -> None:
    backend = (ROOT / "backend/Dockerfile").read_text(encoding="utf-8")
    require(
        "alembic upgrade head && exec uvicorn" in backend and "--workers 1" in backend,
        "Production startup must migrate successfully before a single API worker.",
    )
    require(
        "${FORWARDED_ALLOW_IPS:-127.0.0.1}" in backend,
        "Uvicorn's default proxy allowlist must be loopback, not wildcard.",
    )
    for side in ("backend", "frontend"):
        ignore = (
            (ROOT / side / ".dockerignore").read_text(encoding="utf-8").splitlines()
        )
        require(
            ".env" in ignore and ".env.*" in ignore,
            f"{side} build context must exclude runtime secret files.",
        )
        require("tests" not in ignore, f"{side} test target needs its test sources.")
    frontend = (ROOT / "frontend/Dockerfile").read_text(encoding="utf-8")
    require(
        "RUN npm ci" in frontend and " AS test" in frontend,
        "Frontend must install from its lockfile and expose a Node test target.",
    )
    runtime = frontend.split(" AS production", 1)[-1]
    require(
        "npm" not in runtime and "COPY --from=build /app/dist" in runtime,
        "nginx runtime must serve the built bundle, not execute npm.",
    )
    nginx = (ROOT / "frontend/nginx.conf").read_text(encoding="utf-8")
    required = (
        "server_tokens off;",
        "client_max_body_size 1m;",
        "proxy_connect_timeout 5s;",
        "proxy_send_timeout 30s;",
        "proxy_read_timeout 60s;",
        "proxy_set_header X-Forwarded-For $remote_addr;",
        "proxy_set_header X-Forwarded-Proto $scheme;",
        'proxy_set_header Forwarded "";',
        "https://fonts.googleapis.com",
        "https://fonts.gstatic.com",
        "connect-src 'self'",
        "frame-ancestors 'none'",
        'X-Frame-Options "DENY" always;',
    )
    for fragment in required:
        require(fragment in nginx, "nginx is missing policy fragment: " + fragment)
    require(
        "$proxy_add_x_forwarded_for" not in nginx,
        "nginx must replace, not append, a client-supplied XFF chain.",
    )
    print("PASS static Dockerfile/nginx policy (nginx syntax is a separate CI check)")


def load_yaml(relative: str) -> dict[str, Any]:
    # BaseLoader preserves GitHub Actions' 'on' key instead of YAML 1.1 booleans.
    data = yaml.load(
        (ROOT / relative).read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    require(isinstance(data, dict), relative + " is not a YAML mapping.")
    return dict(data)


def validate_ci_and_render() -> None:
    for side in ("backend", "frontend"):
        workflow = load_yaml(f".github/workflows/{side}-ci.yml")
        for event in ("push", "pull_request"):
            paths = workflow["on"][event]["paths"]
            require(
                "infra/**" in paths and "render.yaml" in paths,
                f"{side} CI must respond to infrastructure and Render changes.",
            )
        require(
            workflow.get("permissions", {}).get("contents") == "read",
            f"{side} CI must use read-only repository permissions.",
        )
    frontend = load_yaml(".github/workflows/frontend-ci.yml")
    require(
        any(step.get("run") == "npm ci" for step in frontend["jobs"]["test"]["steps"]),
        "Frontend CI must use npm ci.",
    )
    backend = load_yaml(".github/workflows/backend-ci.yml")
    steps = backend["jobs"]["test"]["steps"]
    require(
        any(
            "require_integration_results.py" in step.get("run", "")
            and step.get("if") == "always()"
            for step in steps
        ),
        "Required integration tests must fail on missing/empty/skipped reports.",
    )
    require(
        any(
            "-m model" in step.get("run", "")
            and "model-results.xml" in step.get("run", "")
            for step in steps
        )
        and any(
            "require_integration_results.py model-results.xml" in step.get("run", "")
            and step.get("if") == "always()"
            for step in steps
        ),
        "Real pretrained model inference must remain a required, non-skippable CI gate.",
    )
    deployment = load_yaml(".github/workflows/deployment-ci.yml")
    require(
        "http-smoke" in deployment["jobs"],
        "Deployment CI must include real HTTP smoke.",
    )
    for workflow in (frontend, backend, deployment):
        for job in workflow["jobs"].values():
            require(
                1 <= int(job.get("timeout-minutes", "0")) <= 45,
                "Every CI job must have an explicit bounded timeout.",
            )
            for step in job["steps"]:
                require(
                    step.get("continue-on-error") != "true",
                    "Required CI gates must not continue on error.",
                )
    render = load_yaml("render.yaml")
    services = {service["name"]: service for service in render["services"]}
    api = services["gmee-backend"]
    require(
        api.get("healthCheckPath") == "/api/v1/health/ready",
        "Render must probe readiness, not just process liveness.",
    )
    env = {v["key"]: v.get("value") for v in api["envVars"]}
    require(
        env.get("ENABLE_SCHEDULER") == "false"
        and env.get("FORWARDED_ALLOW_IPS") != "*",
        "Render must not automatically run ML jobs or trust arbitrary proxy headers.",
    )
    spa_services = [
        service for service in render["services"] if service.get("runtime") == "static"
    ]
    require(
        len(spa_services) == 1,
        "Render must define exactly one static SPA service.",
    )
    spa = spa_services[0]
    require(
        spa.get("buildCommand") == "npm ci && npm run build",
        "Render static build must install from the lockfile.",
    )
    headers = {v["name"].lower(): v["value"] for v in spa["headers"]}
    csp = headers.get("content-security-policy", "")
    require(
        headers.get("x-frame-options") == "DENY"
        and "frame-ancestors 'none'" in csp
        and "connect-src 'self' https://gmee-backend.onrender.com" in csp,
        "Render CSP must allow its configured API origin and deny framing.",
    )
    print("PASS CI path/strict-test contracts and static Render policy")


def validate_doc_links() -> None:
    documents = [
        ROOT / name for name in ("README.md", "ARCHITECTURE.md", "CHANGELOG.md")
    ]
    documents += sorted((ROOT / "docs").glob("*.md"))
    documents.append(ROOT / "backend/README.md")
    for document in documents:
        for destination in re.findall(
            r"\[[^\]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")
        ):
            target = destination.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            require(
                (document.parent / target).exists(),
                f"Broken documentation link in {document.relative_to(ROOT)}: {target}",
            )
    print("PASS local Markdown file links (anchors and remote URLs not checked)")


def main() -> int:
    try:
        validate_compose()
        validate_images_and_proxy()
        validate_ci_and_render()
        validate_doc_links()
    except (ValidationError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Static deployment validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS deployment static checks; no containers, runtime .env or database accessed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
