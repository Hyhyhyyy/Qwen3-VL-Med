#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_records(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    return value if isinstance(value, list) else [value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--check-images", action="store_true", help="使用 Pillow 检查图片是否可读取")
    parser.add_argument("--issues-jsonl", type=Path, help="把全部数据问题写入 JSONL 记录")
    parser.add_argument("--max-records", type=int, help="只检查前 N 条，用于小样本冒烟验证")
    args = parser.parse_args()

    info_path = args.dataset_dir / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if args.name not in info:
        raise SystemExit(f"dataset_info.json 中未注册数据集：{args.name}")
    data_path = args.dataset_dir / info[args.name]["file_name"]
    records = load_records(data_path)
    if not records:
        raise SystemExit("训练数据为空")
    if args.max_records is not None:
        if args.max_records <= 0:
            raise SystemExit("--max-records 必须大于 0")
        records = records[: args.max_records]

    errors = []
    issue_records = []

    def add_issue(kind: str, index: int, message: str, image: str | None = None) -> None:
        errors.append(message)
        record = {"type": kind, "record_index": index, "message": message}
        if image is not None:
            record["image"] = image
        issue_records.append(record)

    for index, record in enumerate(records, 1):
        messages = record.get("messages", [])
        images = record.get("images", [])
        image_tokens = sum(str(message.get("content", "")).count("<image>") for message in messages)
        if image_tokens != len(images):
            add_issue(
                "image_token_mismatch",
                index,
                f"第 {index} 条：<image> 数量 {image_tokens} != images 数量 {len(images)}",
            )
        for image in images:
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = args.dataset_dir / image_path
            if not image_path.is_file():
                add_issue("image_missing", index, f"第 {index} 条：图片不存在 {image_path}", str(image))
            elif args.check_images:
                try:
                    from PIL import Image

                    with Image.open(image_path) as opened:
                        opened.verify()
                except Exception as error:
                    add_issue(
                        "image_unreadable",
                        index,
                        f"第 {index} 条：图片无法读取 {image_path}（{error}）",
                        str(image),
                    )
        roles = [message.get("role") for message in messages]
        if not messages or roles[0] != "user" or "assistant" not in roles:
            add_issue(
                "invalid_messages",
                index,
                f"第 {index} 条：messages 必须从 user 开始且至少包含一个 assistant 回答",
            )
        for message in messages:
            if message.get("role") == "assistant" and not str(message.get("content", "")).strip():
                add_issue("empty_assistant", index, f"第 {index} 条：assistant 回答为空")

    if args.issues_jsonl:
        args.issues_jsonl.parent.mkdir(parents=True, exist_ok=True)
        temp_path = args.issues_jsonl.with_suffix(args.issues_jsonl.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
            for issue in issue_records:
                stream.write(json.dumps(issue, ensure_ascii=False) + "\n")
        temp_path.replace(args.issues_jsonl)
        print(f"数据问题记录：{args.issues_jsonl}（{len(issue_records)} 条）")

    if errors:
        raise SystemExit("数据校验失败：\n- " + "\n- ".join(errors[:50]))
    suffix = "，图片可读取" if args.check_images else ""
    print(f"数据校验通过：{len(records)} 条，文件 {data_path}{suffix}")
    if len(records) == 1 and "example.ppm" in str(records[0].get("images", [])):
        print("WARNING: 当前仍是示例数据，请在正式训练前替换。")


if __name__ == "__main__":
    main()
