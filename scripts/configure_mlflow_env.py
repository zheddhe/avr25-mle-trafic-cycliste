from __future__ import annotations

import argparse
from pathlib import Path

BLANK_VALUE = '""'

MODE_SOURCES = {
    "compose": {
        "MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI_COMPOSE",
        "MLFLOW_S3_ENDPOINT_URL": "MLFLOW_S3_ENDPOINT_URL_COMPOSE",
        "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID_COMPOSE",
        "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY_COMPOSE",
        "MLFLOW_TRACKING_USERNAME": None,
        "MLFLOW_TRACKING_PASSWORD": None,
    },
    "local": {
        "MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI_LOCAL",
        "MLFLOW_S3_ENDPOINT_URL": None,
        "AWS_ACCESS_KEY_ID": None,
        "AWS_SECRET_ACCESS_KEY": None,
        "MLFLOW_TRACKING_USERNAME": None,
        "MLFLOW_TRACKING_PASSWORD": None,
    },
    "dagshub": {
        "MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI_DAGSHUB",
        "MLFLOW_S3_ENDPOINT_URL": None,
        "AWS_ACCESS_KEY_ID": None,
        "AWS_SECRET_ACCESS_KEY": None,
        "MLFLOW_TRACKING_USERNAME": "MLFLOW_TRACKING_USERNAME_DAGSHUB",
        "MLFLOW_TRACKING_PASSWORD": "MLFLOW_TRACKING_PASSWORD_DAGSHUB",
    },
}


def parse_env_values(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value
    return values


def build_updates(mode: str, values: dict[str, str]) -> dict[str, str]:
    updates: dict[str, str] = {}
    for target_key, source_key in MODE_SOURCES[mode].items():
        if source_key is None:
            updates[target_key] = BLANK_VALUE
            continue
        try:
            updates[target_key] = values[source_key]
        except KeyError as exc:
            raise KeyError(f"Missing required key in env file: {source_key}") from exc
    return updates


def update_env_lines(lines: list[str], updates: dict[str, str]) -> list[str]:
    remaining = dict(updates)
    updated_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            updated_lines.append(line)
            continue

        key, _ = stripped.split("=", 1)
        if key in updates:
            updated_lines.append(f"{key}={updates[key]}\n")
            remaining.pop(key, None)
        else:
            updated_lines.append(line)

    if remaining:
        updated_lines.append("\n# Generated canonical MLflow values\n")
        for key, value in remaining.items():
            updated_lines.append(f"{key}={value}\n")

    return updated_lines


def configure_env_file(env_file: Path, mode: str) -> None:
    lines = env_file.read_text(encoding="utf-8").splitlines(keepends=True)
    values = parse_env_values(lines)
    updates = build_updates(mode, values)
    updated_lines = update_env_lines(lines, updates)
    env_file.write_text("".join(updated_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure canonical MLflow variables in a local env file."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the local env file to update.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_SOURCES),
        required=True,
        help="MLflow mode to apply to canonical variables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_file = Path(args.env_file)
    if not env_file.exists():
        raise FileNotFoundError(f"Env file does not exist: {env_file}")
    configure_env_file(env_file, args.mode)
    print(f"Configured MLflow mode [{args.mode}] in [{env_file}].")


if __name__ == "__main__":
    main()
