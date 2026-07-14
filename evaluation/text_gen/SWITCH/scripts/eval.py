import json
import os
import argparse
import re

CLASS = [
  'action/img2txt','action/img2video','action/video2txt','action/video2video',\
  'final_state/img2txt','final_state/img2img','final_state/video2txt','final_state/video2img',\
  'ui_grounding/img2txt',\
  'verification_action/img2txt', 'verification_action/img2video', 'verification_action/video2txt', 'verification_action/video2video',\
  'verification_state/img2img', 'verification_state/img2txt', 'verification_state/video2img', 'verification_state/video2txt',\
  'vqa_state/img2txt', \
  'vqa_task/video2txt',
]

def load_gt_dict(raw_data):
    """
    Build int(id) -> annotation row for SWITCH vqa.json.
    Supports `data` as a list of dicts with `id`/`GT`, or a JSON object mapping id -> row.
    """
    if isinstance(raw_data, list):
        return {int(str(row["id"])): row for row in raw_data}
    if isinstance(raw_data, dict):
        return {int(str(k)): v for k, v in raw_data.items()}
    raise TypeError("gt `data` must be a list or dict")


def _unwrap_api_results(results, gen_baseline_name):
    if gen_baseline_name in ["qwen3_vl_235b", "qwen3_vl_8b", "claude_sonnet_4"]:
        if isinstance(results, dict) and "choices" in results:
            return results["choices"][0]["message"]["content"]
    return results


def _letter_from_snippet(tail: str) -> str:
    """Parse A–D from a short answer tail (after ### Answer or 'final answer is')."""
    if not tail:
        return ""
    t = tail.strip()
    t = re.sub(r"(?i)only\s+one\s+option", "", t).strip()
    t = re.sub(r"(?i)boxed\s*\{?", "", t).strip()
    mu = t.upper()
    m = re.search(r"\b([A-D])\b", mu)
    if m:
        return m.group(1)
    m = re.search(r"([A-D])", mu)
    if m and m.start() < 8:
        return m.group(1)
    return ""


def _letter_from_prose(text: str) -> str:
    """
    Long-form model output: take the last strong A–D signal (e.g. **A**, **D., 'best option ... D').
    Avoid joining all alphabetic characters from the whole paragraph.
    """
    if not text:
        return ""
    hits = []
    for m in re.finditer(r"\*\*([A-D])\*\*", text, flags=re.I):
        hits.append((m.end(), m.group(1).upper()))
    for m in re.finditer(r"\*\*([A-D])\.", text, flags=re.I):
        hits.append((m.end(), m.group(1).upper()))
    for m in re.finditer(
        r"(?i)\b(?:the\s+)?best\s+(?:option|choice)\s+is\s*:?\s*\*?\*?([A-D])\b",
        text,
    ):
        hits.append((m.end(), m.group(1).upper()))
    for m in re.finditer(r"(?i)\b(?:answer|choice)\s*:?\s*\*?\*?([A-D])\b", text):
        hits.append((m.end(), m.group(1).upper()))
    if hits:
        hits.sort(key=lambda x: x[0])
        return hits[-1][1]
    tail = text[-500:].upper()
    m = re.search(r"\b([A-D])\b", tail)
    if m:
        return m.group(1)
    m = re.search(r"([A-D])\s*$", text.strip().upper())
    if m:
        return m.group(1)
    return ""


def _vlm_original_from_payload(payload, gen_baseline_name) -> str:
    """Full model output string as stored in a result JSON file."""
    if gen_baseline_name in ["qwen3_vl_235b", "qwen3_vl_8b", "claude_sonnet_4"]:
        if isinstance(payload, dict) and "choices" in payload:
            return str(payload["choices"][0]["message"]["content"])
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False)


def _unwrap_results_for_parse(payload, gen_baseline_name):
    """Same mutation as the eval loops: payload -> value passed to parse_mc_prediction."""
    results = payload
    if gen_baseline_name in ["qwen3_vl_235b", "qwen3_vl_8b", "claude_sonnet_4"]:
        if isinstance(results, dict) and "choices" in results:
            results = results["choices"][0]["message"]["content"]
    return results


def _gt_question_text(gt: dict) -> str:
    return str(gt.get("query") or gt.get("question") or "")


def _write_eval_details_json(results_root_dir, gen_baseline_name, rows: list) -> None:
    out = {
        "gen_baseline_name": gen_baseline_name,
        "evaluations": rows,
    }
    path = os.path.join(results_root_dir, "eval_details.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def _load_eval_details_for_dimension(results_root_dir: str, dimension_name: str) -> list:
    path = os.path.join(results_root_dir, "eval_details.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = []
    for row in payload.get("evaluations", []):
        rows.append({
            "dimension_name": dimension_name,
            "id": row.get("id"),
            "question": row.get("question", ""),
            "answer": row.get("vlm-original-answer", ""),
            "extracted_answer": row.get("vlm-extract-answer", ""),
            "correct_answer": row.get("gt-answer", ""),
            "is_match": bool(row.get("match_success", False)),
            "rating": row.get("rating", 0),
        })
    rows.sort(key=lambda r: int(r["id"]) if str(r.get("id", "")).isdigit() else str(r.get("id", "")))
    return rows


def _write_eval_details_all_json(
    baseline_root_dir: str,
    gen_baseline_name: str,
    subcategory_acc: list,
    detail_rows: list,
) -> None:
    os.makedirs(baseline_root_dir, exist_ok=True)
    macro_avg = sum(acc for _, acc in subcategory_acc) / len(subcategory_acc) if subcategory_acc else 0.0
    total_count = len(detail_rows)
    correct_count = sum(1 for row in detail_rows if row["is_match"])
    unmatched_count = sum(1 for row in detail_rows if not row.get("extracted_answer"))
    subcategory_summary = []
    for label, acc in subcategory_acc:
        rows = [row for row in detail_rows if row["dimension_name"] == label]
        subcategory_summary.append({
            "dimension_name": label,
            "accuracy": acc,
            "score_percent": acc * 100,
            "total_count": len(rows),
            "correct_count": sum(1 for row in rows if row["is_match"]),
            "unmatched_count": sum(1 for row in rows if not row.get("extracted_answer")),
        })
    out = {
        "gen_baseline_name": gen_baseline_name,
        "summary": {
            "macro_avg": macro_avg,
            "macro_avg_percent": macro_avg * 100,
            "total_count": total_count,
            "correct_count": correct_count,
            "unmatched_count": unmatched_count,
        },
        "subcategories": subcategory_summary,
        "evaluations": detail_rows,
    }
    path = os.path.join(baseline_root_dir, "eval_details_all.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved detailed evaluation JSON to {path}")


def _iter_prediction_files(results_root_dir):
    """Yield only per-sample prediction JSON files such as 123.json."""
    for file in sorted(os.listdir(results_root_dir)):
        if not file.endswith(".json"):
            continue
        sample_id = os.path.splitext(file)[0]
        if not sample_id.isdigit():
            continue
        yield file, sample_id


def parse_mc_prediction(results, gen_baseline_name) -> str:
    """
    Return a single uppercase letter A–D for multi-choice, or '' if unknown.
    Handles API wrappers, ### Answer / 'The final answer is', and markdown like **A**.
    """
    raw = _unwrap_api_results(results, gen_baseline_name)
    s = str(raw).strip()
    if not s:
        return ""

    if "### Answer" in s:
        after = s.split("### Answer", 1)[1]
        after = after.lstrip("\n").lstrip(":").strip()
        letter = _letter_from_snippet(after)
        if letter:
            return letter

    mfa = re.search(r"(?i)the\s+final\s+answer\s+is\s*:?\s*", s)
    if mfa:
        tail = s[mfa.end() :]
        letter = _letter_from_snippet(tail)
        if letter:
            return letter

    letter = _letter_from_prose(s)
    if letter:
        return letter

    return _letter_from_snippet(s)


def eval_verification_state(gen_baseline_name, results_root_dir, gt_dict):
  correct_count = 0
  unmatched_count = 0
  correct_id = []
  unmatched_id = []
  detail_rows = []
  for file, id in _iter_prediction_files(results_root_dir):
      with open(os.path.join(results_root_dir, file), 'r') as f:
        payload = json.load(f)
      vlm_original = _vlm_original_from_payload(payload, gen_baseline_name)
      results = _unwrap_results_for_parse(payload, gen_baseline_name)
      gt = gt_dict[int(id)]
      compare_result = parse_mc_prediction(results, gen_baseline_name)
      if compare_result == gt['GT']:
        correct_count += 1
        correct_id.append(id)
      if compare_result == '':
        unmatched_count += 1
        unmatched_id.append(id)
      match_success = bool(compare_result and compare_result == gt["GT"])
      detail_rows.append({
        "id": id,
        "question": _gt_question_text(gt),
        "gt-answer": gt["GT"],
        "vlm-original-answer": vlm_original,
        "vlm-extract-answer": compare_result,
        "match_success": match_success,
        "rating": 1 if match_success else 0,
      })
  detail_rows.sort(key=lambda r: int(r["id"]))
  _write_eval_details_json(results_root_dir, gen_baseline_name, detail_rows)
  print(f'{gen_baseline_name} accuracy: {correct_count / len(gt_dict)}')
  print(f'correct_count: {correct_count}')
  print(f'correct_id: {correct_id}')
  print(f'unmatched_count: {unmatched_count}')
  print(f'unmatched_id: {unmatched_id}')
  print(f'total_count: {len(gt_dict)}')
  print(f'accuracy: {correct_count / len(gt_dict)}')
  print("\n\n")
  acc = correct_count / len(gt_dict) if len(gt_dict) else 0.0
  with open(os.path.join(results_root_dir, 'eval_log.txt'), 'w') as f:
    f.write(f'{gen_baseline_name} result: \n')
    f.write(f'correct_count: {correct_count}\n')
    f.write(f'correct_id: {correct_id}\n')
    f.write(f'unmatched_count: {unmatched_count}\n')
    f.write(f'unmatched_id: {unmatched_id}\n')
    f.write(f'total_count: {len(gt_dict)}\n')
    f.write(f'accuracy: {correct_count / len(gt_dict)}\n')
  return acc


def eval_verification_action(gen_baseline_name, results_root_dir, gt_dict):
  correct_count = 0
  unmatched_count = 0
  correct_id = []
  unmatched_id = []
  detail_rows = []
  for file, id in _iter_prediction_files(results_root_dir):
      with open(os.path.join(results_root_dir, file), 'r') as f:
        payload = json.load(f)
      vlm_original = _vlm_original_from_payload(payload, gen_baseline_name)
      results = _unwrap_results_for_parse(payload, gen_baseline_name)
      gt = gt_dict[int(id)]
      compare_result = parse_mc_prediction(results, gen_baseline_name)
      if compare_result == gt['GT']:
        correct_count += 1
        correct_id.append(id)
      if compare_result == '':
        unmatched_count += 1
        unmatched_id.append(id)
      match_success = bool(compare_result and compare_result == gt["GT"])
      detail_rows.append({
        "id": id,
        "question": _gt_question_text(gt),
        "gt-answer": gt["GT"],
        "vlm-original-answer": vlm_original,
        "vlm-extract-answer": compare_result,
        "match_success": match_success,
        "rating": 1 if match_success else 0,
      })
  detail_rows.sort(key=lambda r: int(r["id"]))
  _write_eval_details_json(results_root_dir, gen_baseline_name, detail_rows)
  print(f'{gen_baseline_name} accuracy: {correct_count / len(gt_dict)}')
  print(f'correct_count: {correct_count}')
  print(f'correct_id: {correct_id}')
  print(f'unmatched_count: {unmatched_count}')
  print(f'unmatched_id: {unmatched_id}')
  print(f'total_count: {len(gt_dict)}')
  print(f'accuracy: {correct_count / len(gt_dict)}')
  print("\n\n")
  acc = correct_count / len(gt_dict) if len(gt_dict) else 0.0
  with open(os.path.join(results_root_dir, 'eval_log.txt'), 'w') as f:
    f.write(f'{gen_baseline_name} result: \n')
    f.write(f'correct_count: {correct_count}\n')
    f.write(f'correct_id: {correct_id}\n')
    f.write(f'unmatched_count: {unmatched_count}\n')
    f.write(f'unmatched_id: {unmatched_id}\n')
    f.write(f'total_count: {len(gt_dict)}\n')
    f.write(f'accuracy: {correct_count / len(gt_dict)}\n')
  return acc

def eval(gen_baseline_name, results_root_dir, gt_dict):
    correct_count = 0
    unmatched_count = 0
    correct_id = []
    unmatched_id = []
    detail_rows = []
    for file, id in _iter_prediction_files(results_root_dir):
            with open(os.path.join(results_root_dir, file), 'r') as f:
                payload = json.load(f)
            vlm_original = _vlm_original_from_payload(payload, gen_baseline_name)
            results = _unwrap_results_for_parse(payload, gen_baseline_name)
            gt = gt_dict[int(id)]
            compare_result = parse_mc_prediction(results, gen_baseline_name)
            if compare_result == gt['GT']:
                correct_count += 1
                correct_id.append(id)
            if compare_result == '':
                unmatched_count += 1
                unmatched_id.append(id)
            match_success = bool(compare_result and compare_result == gt["GT"])
            detail_rows.append({
                "id": id,
                "question": _gt_question_text(gt),
                "gt-answer": gt["GT"],
                "vlm-original-answer": vlm_original,
                "vlm-extract-answer": compare_result,
                "match_success": match_success,
                "rating": 1 if match_success else 0,
            })
    detail_rows.sort(key=lambda r: int(r["id"]))
    _write_eval_details_json(results_root_dir, gen_baseline_name, detail_rows)
    print(f'{gen_baseline_name} accuracy: {correct_count / len(gt_dict)}')
    print(f'correct_count: {correct_count}')
    print(f'correct_id: {correct_id}')
    print(f'unmatched_count: {unmatched_count}')
    print(f'unmatched_id: {unmatched_id}')
    print(f'total_count: {len(gt_dict)}')
    print(f'accuracy: {correct_count / len(gt_dict)}')
    print("\n\n")

    acc = correct_count / len(gt_dict) if len(gt_dict) else 0.0
    with open(os.path.join(results_root_dir, 'eval_log.txt'), 'w') as f:
        f.write(f'{gen_baseline_name} result: \n')
        f.write(f'correct_id: {correct_id}\n')
        f.write(f'unmatched_id: {unmatched_id}\n')
        f.write(f'correct_count: {correct_count}\n')
        f.write(f'unmatched_count: {unmatched_count}\n')
        f.write(f'total_count: {len(gt_dict)}\n')
        f.write(f'accuracy: {correct_count / len(gt_dict)}\n')
    return acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_baseline_name', type=str, default='Qwen3_5')
    parser.add_argument('--results_root_dir', type=str, default='results')
    parser.add_argument('--gt_root_dir', type=str, default='your_data_path')
    args = parser.parse_args()

    baseline_root_dir = os.path.join(args.results_root_dir, args.gen_baseline_name)
    subcategory_acc = []
    all_detail_rows = []
    for label in CLASS:
        print(f"Evaluating {label}...")
        results_root_dir = os.path.join(baseline_root_dir, label)
        if not os.path.exists(results_root_dir):
            continue
        gt_root_dir = os.path.join(args.gt_root_dir, label)
        with open(os.path.join(gt_root_dir, 'vqa.json'), 'r') as f:
            gt_dict = load_gt_dict(json.load(f)['data'])
        if "verification_state" in results_root_dir:
            print(results_root_dir)
            acc = eval_verification_state(args.gen_baseline_name, results_root_dir, gt_dict)
        elif "verification_action" in results_root_dir:
            print(results_root_dir)
            acc = eval_verification_action(args.gen_baseline_name, results_root_dir, gt_dict)
        else:
            acc = eval(args.gen_baseline_name, results_root_dir, gt_dict)
        subcategory_acc.append((label, acc))
        all_detail_rows.extend(_load_eval_details_for_dimension(results_root_dir, label))

    print("\n" + "=" * 60)
    print("Per-subcategory accuracy")
    print("=" * 60)
    for label, acc in subcategory_acc:
        print(f"  {label}: {acc:.4f}")
    if subcategory_acc:
        macro = sum(a for _, a in subcategory_acc) / len(subcategory_acc)
        macro *= 100
        print("-" * 60)
        print(f"  Subcategory macro average (macro avg over {len(subcategory_acc)} subcategories): {macro:.2f}")
        _write_eval_details_all_json(
            baseline_root_dir,
            args.gen_baseline_name,
            subcategory_acc,
            all_detail_rows,
        )
    print("=" * 60 + "\n")
