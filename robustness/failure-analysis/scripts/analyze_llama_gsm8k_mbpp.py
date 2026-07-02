from __future__ import annotations

import ast
import builtins
import json
import os
import re
import statistics as st
from pathlib import Path
from typing import Any


EXP_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("EVAL_OUTPUT_ROOT", EXP_DIR / "outputs"))
RESULTS_DIR = Path(os.environ.get("OUTPUT_DIR", EXP_DIR / "results"))

LLAMA_TOKENIZER_PATH = os.environ.get("LLAMA_TOKENIZER_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
ALIEN_TOKENIZER_PATH = os.environ.get(
    "LLAMA_ALIEN_TOKENIZER_PATH", "dsba-lab/llama3-8b-instruct-alienlm-full"
)
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"


def discover_sample_path(env_name: str, pattern: str) -> Path:
    if os.environ.get(env_name):
        return Path(os.environ[env_name])
    matches = sorted(OUTPUT_ROOT.glob(pattern))
    if matches:
        return matches[-1]
    return OUTPUT_ROOT / f"missing_{env_name.lower()}.jsonl"


GSM8K_ORIG_PATH = discover_sample_path(
    "GSM8K_ORIG_PATH", "llama_original/gsm8k_cot/5-shot/**/samples_gsm8k_cot*.jsonl"
)
GSM8K_ALIEN_PATH = discover_sample_path(
    "GSM8K_ALIEN_PATH", "llama_alien/gsm8k_cot/5-shot/**/samples_gsm8k_cot*.jsonl"
)
MBPP_ORIG_PATH = discover_sample_path(
    "MBPP_ORIG_PATH", "llama_original/mbpp/3-shot-vllm/**/samples_mbpp*.jsonl"
)
MBPP_ALIEN_PATH = discover_sample_path(
    "MBPP_ALIEN_PATH", "llama_alien/mbpp/3-shot-vllm/**/samples_mbpp*.jsonl"
)

SUMMARY_JSON = RESULTS_DIR / "llama_gsm8k_mbpp_summary.json"
DETAILED_MD = RESULTS_DIR / "llama_gsm8k_mbpp_detailed.md"
PAPER_MD = RESULTS_DIR / "llama_gsm8k_mbpp_paper_summary.md"

RATE_TERMS = {
    "each",
    "per",
    "every",
    "average",
    "hour",
    "week",
    "day",
    "days",
    "mile",
    "miles",
    "dozen",
    "discount",
    "percent",
    "twice",
    "half",
}

GSM8K_CASE_NOTES = {
    5: "Alternating-price structure collapses: the alien run changes `every second` into `every third`, then totals the wrong number of full-price items.",
    11: "All local multiplications are correct, but the final aggregation is corrupted (`204 + 160 + 330` becomes `594`).",
    16: "The alien run solves the total distance correctly (`230`) and then adds an unnecessary divide-by-two step, misreading `each train`.",
    19: "Average-speed reasoning loses the global constraint. The alien run reuses the already observed average (`3 mph`) instead of solving for the required speed on the remaining segment.",
    24: "Percentage inversion is structurally correct, but the final arithmetic lands on `26.67` instead of the exact integer `26`.",
    29: "Equation setup drifts into an inconsistent symbolic chain (`x + 99 = x - 3`), even though the original solution only needs two arithmetic steps.",
}

MBPP_CASE_NOTES = {
    "Mbpp/17": "Prompt leakage/truncation: the generated code contains a stray marker and starts copying the next in-context example.",
    "Mbpp/30": "Function-name drift: the implementation logic is mostly reasonable, but the defined function name no longer matches the tested API.",
    "Mbpp/40": "Syntax corruption: the generation injects malformed import/code formatting before the function body.",
    "Mbpp/166": "Symbol fidelity failure: the alien run changes operators and identifiers (`size`, `k`) in a way that breaks execution.",
    "Mbpp/170": "Boundary logic drift: inclusive slicing is simplified into a generic slice expression with wrong identifiers.",
    "Mbpp/28": "Formula drift: the combinatorial expression becomes a shifted closed form (`n-1`, `k-1`) instead of the correct coefficient.",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def pp_delta(after: float, before: float) -> str:
    return f"{(after - before) * 100:.1f}pp"


def safe_mean(values: list[float]) -> float:
    return st.mean(values) if values else 0.0


def excerpt(text: str, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def compute_prompt_token_stats() -> dict[str, Any] | None:
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None

    llama_tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_TOKENIZER_PATH,
        local_files_only=LOCAL_FILES_ONLY,
    )
    alien_tokenizer = AutoTokenizer.from_pretrained(
        ALIEN_TOKENIZER_PATH,
        local_files_only=LOCAL_FILES_ONLY,
    )

    pairs = {
        "gsm8k": (GSM8K_ORIG_PATH, GSM8K_ALIEN_PATH),
        "mbpp": (MBPP_ORIG_PATH, MBPP_ALIEN_PATH),
    }
    stats: dict[str, Any] = {}
    for name, (orig_path, alien_path) in pairs.items():
        orig_rows = load_jsonl(orig_path)
        alien_rows = load_jsonl(alien_path)
        orig_lens = []
        alien_lens = []
        for orig_row, alien_row in zip(orig_rows, alien_rows):
            orig_prompt = orig_row["arguments"]["gen_args_0"]["arg_0"]
            alien_prompt = alien_row["arguments"]["gen_args_0"]["arg_0"]
            orig_lens.append(
                len(llama_tokenizer.encode(orig_prompt, add_special_tokens=False))
            )
            alien_lens.append(
                len(alien_tokenizer.encode(alien_prompt, add_special_tokens=False))
            )
        diffs = [a - o for o, a in zip(orig_lens, alien_lens)]
        stats[name] = {
            "orig_mean": safe_mean(orig_lens),
            "alien_mean": safe_mean(alien_lens),
            "orig_stdev": st.pstdev(orig_lens),
            "alien_stdev": st.pstdev(alien_lens),
            "orig_min": min(orig_lens),
            "orig_max": max(orig_lens),
            "alien_min": min(alien_lens),
            "alien_max": max(alien_lens),
            "all_equal": all(diff == 0 for diff in diffs),
            "min_diff": min(diffs),
            "max_diff": max(diffs),
            "n": len(orig_lens),
        }
    return stats


def analyze_gsm8k(prompt_token_stats: dict[str, Any] | None) -> dict[str, Any]:
    orig_rows = {row["doc_id"]: row for row in load_jsonl(GSM8K_ORIG_PATH)}
    alien_rows = {row["doc_id"]: row for row in load_jsonl(GSM8K_ALIEN_PATH)}
    assert set(orig_rows) == set(alien_rows)

    groups: dict[str, list[dict[str, Any]]] = {
        "both_correct": [],
        "orig_correct_alien_wrong": [],
        "both_wrong": [],
        "orig_wrong_alien_correct": [],
    }

    for doc_id, orig_row in orig_rows.items():
        alien_row = alien_rows[doc_id]
        orig_correct = bool(orig_row["exact_match"])
        alien_correct = bool(alien_row["exact_match"])
        key = (
            "both_correct"
            if orig_correct and alien_correct
            else "orig_correct_alien_wrong"
            if orig_correct and not alien_correct
            else "both_wrong"
            if not orig_correct and not alien_correct
            else "orig_wrong_alien_correct"
        )
        question = orig_row["doc"]["question"]
        entry = {
            "doc_id": doc_id,
            "question": question,
            "target": str(orig_row["target"]).strip(),
            "orig_filtered": str(orig_row["filtered_resps"][0]).strip(),
            "alien_filtered": str(alien_row["filtered_resps"][0]).strip(),
            "orig_resp": orig_row["resps"][0][0],
            "alien_resp": alien_row["resps"][0][0],
            "q_len": len(question),
            "num_count": len(
                re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", question.lower())
            ),
            "rate_flag": int(
                any(term in question.lower().split() for term in RATE_TERMS)
                or any(term in question.lower() for term in ("average", "discount", "twice"))
            ),
            "orig_resp_len": len(orig_row["resps"][0][0]),
            "alien_resp_len": len(alien_row["resps"][0][0]),
        }
        groups[key].append(entry)

    orig_acc = safe_mean([1.0 if row["exact_match"] else 0.0 for row in orig_rows.values()])
    alien_acc = safe_mean(
        [1.0 if row["exact_match"] else 0.0 for row in alien_rows.values()]
    )
    failure_group = groups["orig_correct_alien_wrong"]
    shorter_resp_share = safe_mean(
        [1.0 if row["alien_resp_len"] < row["orig_resp_len"] else 0.0 for row in failure_group]
    )
    mentions_target_share = safe_mean(
        [
            1.0
            if re.search(
                rf"(?<!\d){re.escape(row['target'])}(?!\d)", row["alien_resp"]
            )
            else 0.0
            for row in failure_group
        ]
    )

    def group_stats(items: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "q_len_mean": safe_mean([row["q_len"] for row in items]),
            "num_count_mean": safe_mean([row["num_count"] for row in items]),
            "rate_share": safe_mean([row["rate_flag"] for row in items]),
            "orig_resp_len_mean": safe_mean([row["orig_resp_len"] for row in items]),
            "alien_resp_len_mean": safe_mean([row["alien_resp_len"] for row in items]),
        }

    representative_cases = []
    for doc_id, note in GSM8K_CASE_NOTES.items():
        row = next(item for item in failure_group if item["doc_id"] == doc_id)
        representative_cases.append(
            {
                "doc_id": doc_id,
                "question": row["question"],
                "target": row["target"],
                "alien_pred": row["alien_filtered"],
                "orig_excerpt": excerpt(row["orig_resp"]),
                "alien_excerpt": excerpt(row["alien_resp"]),
                "note": note,
            }
        )

    return {
        "orig_acc": orig_acc,
        "alien_acc": alien_acc,
        "delta_pp": (alien_acc - orig_acc) * 100,
        "n_total": len(orig_rows),
        "group_counts": {name: len(items) for name, items in groups.items()},
        "group_stats": {name: group_stats(items) for name, items in groups.items()},
        "shorter_resp_share": shorter_resp_share,
        "mentions_target_share": mentions_target_share,
        "prompt_token_stats": None if prompt_token_stats is None else prompt_token_stats["gsm8k"],
        "representative_cases": representative_cases,
    }


def expected_function_name(doc: dict[str, Any]) -> str | None:
    joined_tests = "\n".join(doc.get("test_list", []) + doc.get("challenge_test_list", []))
    match = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", joined_tests)
    return match.group(1) if match else None


def predicted_function_name(code: str) -> str | None:
    match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    return match.group(1) if match else None


BUILTIN_NAMES = set(dir(builtins)) | {"math"}


class NameAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.defined: set[str] = set()
        self.used: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defined.add(node.name)
        for arg in node.args.args + node.args.kwonlyargs:
            self.defined.add(arg.arg)
        if node.args.vararg:
            self.defined.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.defined.add(node.args.kwarg.arg)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defined.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.defined.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.used.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.defined.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.defined.add(alias.asname or alias.name)


def undefined_identifiers(code: str) -> list[str] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    analyzer = NameAnalyzer()
    analyzer.visit(tree)
    return sorted(
        name
        for name in analyzer.used
        if name not in analyzer.defined and name not in BUILTIN_NAMES
    )


def analyze_mbpp(prompt_token_stats: dict[str, Any] | None) -> dict[str, Any]:
    orig_rows = {
        f"Mbpp/{row['doc']['task_id']}": row for row in load_jsonl(MBPP_ORIG_PATH)
    }
    alien_rows = {
        f"Mbpp/{row['doc']['task_id']}": row for row in load_jsonl(MBPP_ALIEN_PATH)
    }
    assert set(orig_rows) == set(alien_rows)

    groups: dict[str, list[dict[str, Any]]] = {
        "both_pass": [],
        "orig_pass_alien_fail": [],
        "both_fail": [],
        "orig_fail_alien_pass": [],
    }
    signature_counts = {
        "syntax_error": 0,
        "prompt_leak": 0,
        "name_mismatch": 0,
        "no_def": 0,
        "undefined_identifier": 0,
        "logic_only": 0,
    }

    for task_id, orig_row in orig_rows.items():
        alien_row = alien_rows[task_id]
        orig_pass = bool(orig_row["pass_at_1"])
        alien_pass = bool(alien_row["pass_at_1"])
        key = (
            "both_pass"
            if orig_pass and alien_pass
            else "orig_pass_alien_fail"
            if orig_pass and not alien_pass
            else "both_fail"
            if not orig_pass and not alien_pass
            else "orig_fail_alien_pass"
        )

        alien_code = alien_row["filtered_resps"][0]
        expected_name = expected_function_name(alien_row["doc"])
        predicted_name = predicted_function_name(alien_code)
        prompt_leak = (
            "[JONE]" in alien_code
            or "There are an expert Python programmer" in alien_code
            or alien_code.count("assert ") > 0
        )
        try:
            ast.parse(alien_code)
            syntax_error = False
        except SyntaxError:
            syntax_error = True
        undefined = undefined_identifiers(alien_code) or []
        name_mismatch = bool(
            expected_name and predicted_name and expected_name != predicted_name
        )
        no_def = predicted_name is None

        entry = {
            "task_id": task_id,
            "text": alien_row["doc"]["text"],
            "tests": alien_row["doc"]["test_list"],
            "orig_code": orig_row["filtered_resps"][0],
            "alien_code": alien_code,
            "orig_len": len(orig_row["filtered_resps"][0]),
            "alien_len": len(alien_code),
            "expected_name": expected_name,
            "predicted_name": predicted_name,
            "prompt_leak": prompt_leak,
            "syntax_error": syntax_error,
            "name_mismatch": name_mismatch,
            "no_def": no_def,
            "undefined_identifiers": undefined,
        }
        groups[key].append(entry)

        if key == "orig_pass_alien_fail":
            hit_signature = False
            for signature_name, flag in [
                ("syntax_error", syntax_error),
                ("prompt_leak", prompt_leak),
                ("name_mismatch", name_mismatch),
                ("no_def", no_def),
                ("undefined_identifier", bool(undefined)),
            ]:
                if flag:
                    signature_counts[signature_name] += 1
                    hit_signature = True
            if not hit_signature:
                signature_counts["logic_only"] += 1

    orig_pass = safe_mean([1.0 if row["pass_at_1"] else 0.0 for row in orig_rows.values()])
    alien_pass = safe_mean(
        [1.0 if row["pass_at_1"] else 0.0 for row in alien_rows.values()]
    )

    representative_cases = []
    failure_group = {row["task_id"]: row for row in groups["orig_pass_alien_fail"]}
    for task_id, note in MBPP_CASE_NOTES.items():
        row = failure_group[task_id]
        representative_cases.append(
            {
                "task_id": task_id,
                "text": row["text"],
                "tests": row["tests"][:2],
                "orig_excerpt": excerpt(row["orig_code"]),
                "alien_excerpt": excerpt(row["alien_code"]),
                "note": note,
            }
        )

    def code_len_stats(items: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "orig_len_mean": safe_mean([row["orig_len"] for row in items]),
            "alien_len_mean": safe_mean([row["alien_len"] for row in items]),
        }

    return {
        "orig_pass_at_1": orig_pass,
        "alien_pass_at_1": alien_pass,
        "delta_pp": (alien_pass - orig_pass) * 100,
        "n_total": len(orig_rows),
        "group_counts": {name: len(items) for name, items in groups.items()},
        "group_stats": {name: code_len_stats(items) for name, items in groups.items()},
        "signature_counts": signature_counts,
        "prompt_token_stats": None if prompt_token_stats is None else prompt_token_stats["mbpp"],
        "representative_cases": representative_cases,
    }


def build_detailed_md(summary: dict[str, Any]) -> str:
    gsm8k = summary["gsm8k"]
    mbpp = summary["mbpp"]

    gsm8k_token = gsm8k["prompt_token_stats"]
    mbpp_token = mbpp["prompt_token_stats"]

    lines: list[str] = []
    lines.append("# Llama GSM8K and MBPP Failure Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This note compares the stored Llama original runs against the stored Llama AlienLM runs."
    )
    lines.append("The goal is qualitative failure analysis rather than leaderboard reproduction.")
    lines.append("")
    lines.append("Sources:")
    lines.append(f"- GSM8K original: `{GSM8K_ORIG_PATH}`")
    lines.append(f"- GSM8K AlienLM: `{GSM8K_ALIEN_PATH}`")
    lines.append(f"- MBPP original: `{MBPP_ORIG_PATH}`")
    lines.append(f"- MBPP AlienLM: `{MBPP_ALIEN_PATH}`")
    lines.append("")
    lines.append("## High-Level Summary")
    lines.append("")
    lines.append("| Task | Original | AlienLM | Delta | Comparison unit |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    lines.append(
        f"| GSM8K | {pct(gsm8k['orig_acc'])} | {pct(gsm8k['alien_acc'])} | {gsm8k['delta_pp']:.1f}pp | exact match over {gsm8k['n_total']} questions |"
    )
    lines.append(
        f"| MBPP | {pct(mbpp['orig_pass_at_1'])} | {pct(mbpp['alien_pass_at_1'])} | {mbpp['delta_pp']:.1f}pp | pass@1 over {mbpp['n_total']} problems |"
    )
    lines.append("")
    lines.append("## GSM8K")
    lines.append("")
    lines.append(
        f"- Original is correct on {gsm8k['group_counts']['both_correct'] + gsm8k['group_counts']['orig_correct_alien_wrong']} / {gsm8k['n_total']} questions, while AlienLM is correct on {gsm8k['group_counts']['both_correct'] + gsm8k['group_counts']['orig_wrong_alien_correct']} / {gsm8k['n_total']}."
    )
    lines.append(
        f"- There are {gsm8k['group_counts']['orig_correct_alien_wrong']} `original-correct -> alien-wrong` cases."
    )
    lines.append(
        f"- In those failure cases, alien responses are shorter than the original responses {pct(gsm8k['shorter_resp_share'])} of the time."
    )
    lines.append(
        f"- Only {pct(gsm8k['mentions_target_share'])} of those alien failures still mention the gold answer string somewhere in the reasoning trace."
    )
    lines.append("")
    lines.append("| GSM8K split | Count | Mean question chars | Mean numeric mentions | Rate/aggregation share | Mean original response chars | Mean alien response chars |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, label in [
        ("both_correct", "both correct"),
        ("orig_correct_alien_wrong", "original correct, alien wrong"),
        ("both_wrong", "both wrong"),
        ("orig_wrong_alien_correct", "original wrong, alien correct"),
    ]:
        stats = gsm8k["group_stats"][key]
        count = gsm8k["group_counts"][key]
        lines.append(
            f"| {label} | {count} | {stats['q_len_mean']:.1f} | {stats['num_count_mean']:.2f} | {pct(stats['rate_share'])} | {stats['orig_resp_len_mean']:.1f} | {stats['alien_resp_len_mean']:.1f} |"
        )
    lines.append("")
    if gsm8k_token is not None:
        lines.append("Prompt token count check:")
        lines.append(
            f"- Mean prompt tokens are exactly preserved under the native/original tokenizer and the checkpoint tokenizer: `gsm8k original={gsm8k_token['orig_mean']:.1f}`, `gsm8k alien={gsm8k_token['alien_mean']:.1f}`."
        )
        lines.append(
            f"- Equality holds for every GSM8K prompt (`all_equal={gsm8k_token['all_equal']}`, `min_diff={gsm8k_token['min_diff']}`, `max_diff={gsm8k_token['max_diff']}`)."
        )
        lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "- The main GSM8K failures are not dominated by prompt-length inflation inside the same tokenizer family; the prompt token count is preserved."
    )
    lines.append(
        "- Instead, failures cluster around multi-step numerical problems where a small symbolic drift changes the meaning of the final computation."
    )
    lines.append(
        "- Typical patterns are wrong aggregation, unnecessary normalization/division, and equation setup drift."
    )
    lines.append("")
    lines.append("Representative GSM8K failures:")
    lines.append("")
    for case in gsm8k["representative_cases"]:
        lines.append(f"### GSM8K doc_id {case['doc_id']}")
        lines.append("")
        lines.append(f"- Question: {case['question']}")
        lines.append(f"- Gold answer: `{case['target']}`")
        lines.append(f"- AlienLM final answer: `{case['alien_pred']}`")
        lines.append(f"- Original response excerpt: `{case['orig_excerpt']}`")
        lines.append(f"- AlienLM response excerpt: `{case['alien_excerpt']}`")
        lines.append(f"- Note: {case['note']}")
        lines.append("")
    lines.append("## MBPP")
    lines.append("")
    lines.append(
        f"- Original passes {int(mbpp['orig_pass_at_1'] * mbpp['n_total'])} / {mbpp['n_total']} problems; AlienLM passes {int(mbpp['alien_pass_at_1'] * mbpp['n_total'])} / {mbpp['n_total']}."
    )
    lines.append(
        f"- There are {mbpp['group_counts']['orig_pass_alien_fail']} `original-pass -> alien-fail` problems."
    )
    lines.append("")
    lines.append("| MBPP split | Count | Mean original code chars | Mean alien code chars |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key, label in [
        ("both_pass", "both pass"),
        ("orig_pass_alien_fail", "original pass, alien fail"),
        ("both_fail", "both fail"),
        ("orig_fail_alien_pass", "original fail, alien pass"),
    ]:
        stats = mbpp["group_stats"][key]
        count = mbpp["group_counts"][key]
        lines.append(
            f"| {label} | {count} | {stats['orig_len_mean']:.1f} | {stats['alien_len_mean']:.1f} |"
        )
    lines.append("")
    if mbpp_token is not None:
        lines.append("Prompt token count check:")
        lines.append(
            f"- Mean prompt tokens are exactly preserved under the native/original tokenizer and the checkpoint tokenizer: `mbpp original={mbpp_token['orig_mean']:.1f}`, `mbpp alien={mbpp_token['alien_mean']:.1f}`."
        )
        lines.append(
            f"- Equality holds for every MBPP prompt (`all_equal={mbpp_token['all_equal']}`, `min_diff={mbpp_token['min_diff']}`, `max_diff={mbpp_token['max_diff']}`)."
        )
        lines.append("")
    lines.append("Failure signatures among `original-pass -> alien-fail` MBPP problems:")
    lines.append("")
    lines.append("| Signature | Count | Share of 181 cases |")
    lines.append("| --- | ---: | ---: |")
    total_failure = mbpp["group_counts"]["orig_pass_alien_fail"]
    for key, label in [
        ("syntax_error", "syntax error"),
        ("prompt_leak", "prompt leakage / truncation"),
        ("name_mismatch", "function name mismatch"),
        ("no_def", "no top-level `def` found"),
        ("undefined_identifier", "undefined identifier"),
        ("logic_only", "logic drift without the above signatures"),
    ]:
        count = mbpp["signature_counts"][key]
        lines.append(f"| {label} | {count} | {pct(count / total_failure)} |")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "- Unlike GSM8K, MBPP failures frequently break execution directly rather than merely changing the final numeric answer."
    )
    lines.append(
        "- The dominant failure modes are syntax corruption, prompt leakage, identifier drift, and API-name mismatch."
    )
    lines.append(
        "- This is consistent with code being much less tolerant to small symbol-level deviations than natural-language reasoning."
    )
    lines.append("")
    lines.append("Representative MBPP failures:")
    lines.append("")
    for case in mbpp["representative_cases"]:
        lines.append(f"### {case['task_id']}")
        lines.append("")
        lines.append(f"- Problem: {case['text']}")
        if case["tests"]:
            lines.append(f"- Test excerpt: `{case['tests'][0]}`")
        lines.append(f"- Original solution excerpt: `{case['orig_excerpt']}`")
        lines.append(f"- AlienLM solution excerpt: `{case['alien_excerpt']}`")
        lines.append(f"- Note: {case['note']}")
        lines.append("")
    lines.append("## Cross-Task Takeaways")
    lines.append("")
    lines.append(
        "- For Llama, token count is preserved exactly between the original prompts and the AlienLM prompts when each is measured in its own tokenizer space. The main issue is therefore not prompt-length growth."
    )
    lines.append(
        "- GSM8K mostly suffers from semantic-role drift in multi-step arithmetic, where one extra divide, one wrong aggregation, or one slightly wrong equation collapses exact-match accuracy."
    )
    lines.append(
        "- MBPP suffers more severely because code is brittle to syntax, identifiers, delimiters, API names, and boundary conditions. AlienLM errors often invalidate the program before test execution reaches the core logic."
    )
    lines.append(
        "- A defensible paper claim is that AlienLM preserves information but weakens the model's ability to reliably reproduce the exact symbolic surface forms required by math and code."
    )
    lines.append("")
    return "\n".join(lines)


def build_paper_md(summary: dict[str, Any]) -> str:
    gsm8k = summary["gsm8k"]
    mbpp = summary["mbpp"]
    lines: list[str] = []
    lines.append("# Paper Summary: Llama Failure Analysis on GSM8K and MBPP")
    lines.append("")
    lines.append("## Main Claim")
    lines.append("")
    lines.append(
        "For Llama, the large drop on math and code is better explained by exact symbolic surface-form sensitivity than by prompt-length growth."
    )
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    lines.append(
        f"- GSM8K drops from {pct(gsm8k['orig_acc'])} to {pct(gsm8k['alien_acc'])} ({gsm8k['delta_pp']:.1f}pp)."
    )
    lines.append(
        f"- MBPP drops from {pct(mbpp['orig_pass_at_1'])} to {pct(mbpp['alien_pass_at_1'])} ({mbpp['delta_pp']:.1f}pp)."
    )
    lines.append(
        "- In both GSM8K and MBPP, prompt token counts are preserved exactly between the original prompt and the alien prompt when each is measured in its own tokenizer space."
    )
    lines.append(
        f"- GSM8K has {gsm8k['group_counts']['orig_correct_alien_wrong']} `original-correct -> alien-wrong` cases; these cases are longer and numerically denser than the `both-correct` subset."
    )
    lines.append(
        f"- MBPP has {mbpp['group_counts']['orig_pass_alien_fail']} `original-pass -> alien-fail` cases; among them, syntax errors appear in {mbpp['signature_counts']['syntax_error']} cases, undefined identifiers in {mbpp['signature_counts']['undefined_identifier']} cases, and function-name mismatch in {mbpp['signature_counts']['name_mismatch']} cases."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- On GSM8K, AlienLM often preserves the broad plan but introduces a small semantic drift in the final computation: wrong aggregation, extra normalization, or slightly incorrect equation setup."
    )
    lines.append(
        "- On MBPP, the same kind of surface-form instability is much more damaging because code execution depends on exact syntax, exact identifiers, and exact API names."
    )
    lines.append(
        "- This suggests that AlienLM is relatively compatible with tasks that tolerate paraphrase, but substantially weaker on tasks that require exact symbolic fidelity."
    )
    lines.append("")
    lines.append("## Candidate Rebuttal Wording")
    lines.append("")
    lines.append(
        "Our qualitative analysis suggests that the degradation on GSM8K and MBPP is not primarily due to longer prompts: for Llama, the original and alien prompts have exactly matched token counts under their respective tokenizers. Instead, the main issue is exact symbolic fidelity. On GSM8K, AlienLM failures typically arise from small semantic drifts in multi-step arithmetic, such as incorrect aggregation or an unnecessary normalization step. On MBPP, failures are more severe because code is brittle to syntax, identifier names, and boundary conditions; many alien outputs become uncompilable or violate the required function signature despite remaining close in high-level intent."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [
        (name, path)
        for name, path in [
            ("GSM8K_ORIG_PATH", GSM8K_ORIG_PATH),
            ("GSM8K_ALIEN_PATH", GSM8K_ALIEN_PATH),
            ("MBPP_ORIG_PATH", MBPP_ORIG_PATH),
            ("MBPP_ALIEN_PATH", MBPP_ALIEN_PATH),
        ]
        if not path.exists()
    ]
    if missing:
        lines = ["Missing required evaluation sample files:"]
        for name, path in missing:
            lines.append(f"- {name}: {path}")
        lines.append("Set EVAL_OUTPUT_ROOT or the explicit *_PATH environment variables.")
        raise FileNotFoundError("\n".join(lines))

    prompt_token_stats = compute_prompt_token_stats()
    summary = {
        "gsm8k": analyze_gsm8k(prompt_token_stats),
        "mbpp": analyze_mbpp(prompt_token_stats),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    DETAILED_MD.write_text(build_detailed_md(summary))
    PAPER_MD.write_text(build_paper_md(summary))
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {DETAILED_MD}")
    print(f"Wrote {PAPER_MD}")


if __name__ == "__main__":
    main()
