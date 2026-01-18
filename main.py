import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import matplotlib.pyplot as plt

from gemini_compat import OpenAICompat


load_dotenv()


MODEL_DEFAULT = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MODEL_SOLVER_1 = os.getenv("GEMINI_MODEL_SOLVER_1", MODEL_DEFAULT)
MODEL_SOLVER_2 = os.getenv("GEMINI_MODEL_SOLVER_2", MODEL_DEFAULT)
MODEL_SOLVER_3 = os.getenv("GEMINI_MODEL_SOLVER_3", MODEL_DEFAULT)
MODEL_JUDGE = os.getenv("GEMINI_MODEL_JUDGE", MODEL_DEFAULT)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SOLVER_PROFILES = {
    "solver_1": {
        "style": "Use rigorous step-by-step derivations and verify with a quick sanity check.",
        "solve_temp": 0.2,
        "review_temp": 0.2,
        "refine_temp": 0.2,
    },
    "solver_2": {
        "style": "Actively look for edge cases and alternative approaches; be skeptical of leaps.",
        "solve_temp": 0.5,
        "review_temp": 0.4,
        "refine_temp": 0.3,
    },
    "solver_3": {
        "style": "Use heuristic reasoning first, then verify with explicit calculation.",
        "solve_temp": 0.7,
        "review_temp": 0.5,
        "refine_temp": 0.4,
    },
}


@dataclass
class Problem:
    id: str
    category: str
    question: str
    ground_truth: Any
    answer_type: str  # "number", "text", "set"


def _normalize_text(text: str) -> str:
    s = str(text or "").lower()
    s = re.sub(r"[^a-z0-9\.\-\/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_number(text: str) -> Optional[float]:
    if text is None:
        return None
    s = str(text)
    frac = re.search(r"(-?\d+(?:\.\d+)?)[\s]*/[\s]*(\d+(?:\.\d+)?)", s)
    if frac:
        try:
            num = float(frac.group(1))
            den = float(frac.group(2))
            if den != 0:
                return num / den
        except Exception:
            pass
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _normalize_set(text: str) -> List[str]:
    s = _normalize_text(text)
    parts = re.split(r"\band\b|;|,", s)
    cleaned = []
    for p in parts:
        t = p.strip()
        if t:
            cleaned.append(t)
    return sorted(set(cleaned))


def is_correct(final_answer: str, expected: Any, answer_type: str) -> bool:
    if answer_type == "number":
        pred = _extract_number(final_answer)
        if pred is None:
            return False
        exp = float(expected)
        tol = max(1e-6, 1e-2 * max(1.0, abs(exp)))
        return abs(pred - exp) <= tol
    if answer_type == "set":
        pred_set = _normalize_set(final_answer)
        exp_set = sorted(set(_normalize_set(expected)))
        return pred_set == exp_set
    return _normalize_text(final_answer) == _normalize_text(expected)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    max_tokens=5000
    client = OpenAICompat(default_model=model)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )
    return resp.choices[0].message.content.strip() if resp.choices else ""


def safe_json_loads(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{(?:.|\n)*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def extract_critiques(reviews: List[Dict[str, Any]]) -> List[str]:
    critiques: List[str] = []
    for r in reviews:
        ev = r.get("evaluation", {}) if isinstance(r, dict) else {}
        for w in ev.get("weaknesses", []) or []:
            critiques.append(str(w))
        for s in ev.get("suggested_changes", []) or []:
            critiques.append(str(s))
        for e in ev.get("errors", []) or []:
            if isinstance(e, dict):
                loc = e.get("location", "")
                desc = e.get("description", "")
                et = e.get("error_type", "")
                sev = e.get("severity", "")
                details = " | ".join([p for p in [loc, et, sev, desc] if p])
                critiques.append(details or str(e))
            else:
                critiques.append(str(e))
    # de-dup while preserving order
    seen = set()
    ordered = []
    for c in critiques:
        key = c.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def stage0_self_assess(problem: Problem, llm_id: str) -> Dict[str, Any]:
    system_prompt = "You are an LLM self-assessing role fit for a collaborative debate system."
    user_prompt = (
        "Given the problem below, decide which role fits you best for this question.\n"
        "Return ONLY a JSON object with keys: role_preferences (list), confidence_by_role (object), reasoning (string).\n"
        "Use confidence values between 0 and 1. Include Judge confidence with key 'Judge'.\n"
        "For solver confidence, use a key that starts with 'Solver' (e.g., 'Solver_1').\n"
        f"\nProblem:\n{problem.question}"
    )
    raw = call_llm(system_prompt, user_prompt, MODEL_DEFAULT, 0.2, 800)
    data = safe_json_loads(raw)
    data.setdefault("role_preferences", ["Solver", "Judge"])
    data.setdefault("confidence_by_role", {"Solver_1": 0.5, "Judge": 0.5})
    data.setdefault("reasoning", "No reasoning provided.")
    data["llm_id"] = llm_id
    return data


def _solver_confidence(conf: Dict[str, Any]) -> float:
    for k, v in conf.items():
        if str(k).lower().startswith("solver"):
            try:
                return float(v)
            except Exception:
                return 0.0
    return 0.0


def assign_roles(self_assessments: List[Dict[str, Any]]) -> Dict[str, str]:
    judge_candidate = max(
        self_assessments,
        key=lambda d: float(d.get("confidence_by_role", {}).get("Judge", 0.0)),
    )
    remaining = [d for d in self_assessments if d["llm_id"] != judge_candidate["llm_id"]]
    remaining_sorted = sorted(
        remaining, key=lambda d: _solver_confidence(d.get("confidence_by_role", {})), reverse=True
    )
    role_map = {
        "Judge": judge_candidate["llm_id"],
        "Solver_1": remaining_sorted[0]["llm_id"],
        "Solver_2": remaining_sorted[1]["llm_id"],
        "Solver_3": remaining_sorted[2]["llm_id"],
    }
    return role_map


def stage1_solve(
    problem: Problem,
    solver_label: str,
    agent_id: str,
    model: str,
    temperature: float,
    style_guidance: str,
) -> Dict[str, Any]:
    system_prompt = (
        f"You are {solver_label} (agent {agent_id}) in a multi-LLM debate system.\n"
        f"Style guidance: {style_guidance}"
    )
    user_prompt = (
        "Solve the problem independently. Provide step-by-step reasoning and a final concise answer.\n"
        "Return ONLY JSON with keys: solution (step-by-step), final_answer.\n"
        f"\nProblem:\n{problem.question}"
    )
    raw = call_llm(system_prompt, user_prompt, model, temperature, 1200)
    data = safe_json_loads(raw)
    data.setdefault("solution", raw)
    data.setdefault("final_answer", "")
    return data


def stage2_peer_review(
    problem: Problem,
    reviewer_label: str,
    reviewer_agent_id: str,
    target_label: str,
    target_solution: Dict[str, Any],
    model: str,
    temperature: float,
    style_guidance: str,
) -> Dict[str, Any]:
    system_prompt = (
        f"You are {reviewer_label} (agent {reviewer_agent_id}), reviewing {target_label}'s solution.\n"
        f"Style guidance: {style_guidance}"
    )
    user_prompt = (
        "Evaluate the solution critically. Return ONLY JSON with structure:\n"
        "{\n"
        '  "solution_id": "solver_X",\n'
        '  "evaluation": {"strengths":[], "weaknesses":[], "errors":[{"location":"","error_type":"","description":"","severity":"minor|major|critical"}], "suggested_changes":[]},\n'
        '  "overall_assessment": "promising_but_flawed|strong|flawed|uncertain"\n'
        "}\n"
        f"\nProblem:\n{problem.question}\n\nSolution:\n{json.dumps(target_solution, ensure_ascii=False)}"
    )
    raw = call_llm(system_prompt, user_prompt, model, temperature, 1200)
    data = safe_json_loads(raw)
    data.setdefault("solution_id", target_label.lower())
    data.setdefault("evaluation", {"strengths": [], "weaknesses": [], "errors": [], "suggested_changes": []})
    errors = data["evaluation"].get("errors")
    if not isinstance(errors, list):
        errors = []
    normalized_errors = []
    for e in errors:
        if isinstance(e, dict):
            normalized_errors.append(
                {
                    "location": str(e.get("location", "")),
                    "error_type": str(e.get("error_type", "")),
                    "description": str(e.get("description", "")),
                    "severity": str(e.get("severity", "")),
                }
            )
        else:
            normalized_errors.append(
                {
                    "location": "",
                    "error_type": "",
                    "description": str(e),
                    "severity": "",
                }
            )
    data["evaluation"]["errors"] = normalized_errors
    data.setdefault("overall_assessment", "uncertain")
    return data


def stage3_refine(
    problem: Problem,
    solver_label: str,
    agent_id: str,
    original_solution: Dict[str, Any],
    reviews: List[Dict[str, Any]],
    model: str,
    temperature: float,
    style_guidance: str,
) -> Dict[str, Any]:
    system_prompt = (
        f"You are {solver_label} (agent {agent_id}). Refine your solution based on peer reviews.\n"
        f"Style guidance: {style_guidance}"
    )
    critique_list = extract_critiques(reviews)
    user_prompt = (
        "Revise your solution. Address each critique explicitly or defend if incorrect.\n"
        "Return ONLY JSON with keys: changes_made (list), refined_solution, refined_answer, confidence.\n"
        "Each item in changes_made must be: {\"critique\":\"...\",\"response\":\"...\",\"accepted\":true|false}.\n"
        f"Critiques to address:\n{json.dumps(critique_list, ensure_ascii=False)}\n"
        f"\nProblem:\n{problem.question}\n\nOriginal:\n{json.dumps(original_solution, ensure_ascii=False)}"
        f"\n\nReviews:\n{json.dumps(reviews, ensure_ascii=False)}"
    )
    raw = call_llm(system_prompt, user_prompt, model, temperature, 1400)
    data = safe_json_loads(raw)
    data.setdefault("changes_made", [])
    if not isinstance(data.get("changes_made"), list) or not data["changes_made"]:
        data["changes_made"] = [
            {"critique": c, "response": "No response provided.", "accepted": False} for c in critique_list
        ]
    data.setdefault("refined_solution", raw)
    data.setdefault("refined_answer", original_solution.get("final_answer", ""))
    data.setdefault("confidence", 0.5)
    return data


def stage4_judge(
    problem: Problem,
    judge_agent_id: str,
    original_solutions: Dict[str, Any],
    reviews: List[Dict[str, Any]],
    refined: Dict[str, Any],
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    system_prompt = (
        f"You are the Final Judge (agent {judge_agent_id}). Select the best refined solution.\n"
        "Do NOT solve the problem from scratch. Compare the provided solutions only."
    )
    user_prompt = (
        "Choose the best final solution. Return ONLY JSON with keys: winner, confidence, reasoning.\n"
        f"\nProblem:\n{problem.question}\n"
        f"\nOriginal Solutions:\n{json.dumps(original_solutions, ensure_ascii=False)}"
        f"\n\nPeer Reviews:\n{json.dumps(reviews, ensure_ascii=False)}"
        f"\n\nRefined Solutions:\n{json.dumps(refined, ensure_ascii=False)}"
    )
    raw = call_llm(system_prompt, user_prompt, model, temperature, 1200)
    data = safe_json_loads(raw)
    data.setdefault("winner", "solver_1")
    data.setdefault("confidence", 0.5)
    data.setdefault("reasoning", "No reasoning provided.")
    return data


def build_dataset() -> List[Problem]:
    return [
        Problem(
            "P01",
            "Mathematical/Logical Reasoning",
            "In how many ways can you tile a 3x8 rectangle with 2x1 dominoes?",
            153,
            "number",
        ),
        Problem(
            "P10",
            "Physics & Scientific Reasoning",
            "A 10m ladder leans against a frictionless wall at 60 degrees from the ground. What is the minimum coefficient of static friction at the ground to prevent slipping? (Use g=10 m/s^2)",
            0.289,
            "number",
        ),
        Problem(
            "P23",
            "Strategic Game Theory",
            "Cournot competition: P=100-Q, 3 firms, MC=10 each. What is equilibrium quantity for each firm?",
            22.5,
            "number",
        ),
    ]


def run_pipeline(problem: Problem) -> Dict[str, Any]:
    llm_ids = ["llm_1", "llm_2", "llm_3", "llm_4"]
    stage0 = [stage0_self_assess(problem, llm_id) for llm_id in llm_ids]
    role_map = assign_roles(stage0)
    role_to_agent = {
        "solver_1": role_map["Solver_1"],
        "solver_2": role_map["Solver_2"],
        "solver_3": role_map["Solver_3"],
        "judge": role_map["Judge"],
    }

    solver_models = {
        "solver_1": MODEL_SOLVER_1,
        "solver_2": MODEL_SOLVER_2,
        "solver_3": MODEL_SOLVER_3,
    }

    default_profile = {
        "style": "Provide clear step-by-step reasoning and a concise final answer.",
        "solve_temp": 0.3,
        "review_temp": 0.3,
        "refine_temp": 0.3,
    }

    stage1 = {}
    for solver_label in ["solver_1", "solver_2", "solver_3"]:
        profile = SOLVER_PROFILES.get(solver_label, default_profile)
        stage1[solver_label] = stage1_solve(
            problem,
            solver_label,
            role_to_agent[solver_label],
            solver_models[solver_label],
            profile["solve_temp"],
            profile["style"],
        )

    stage2 = []
    pairs = [
        ("solver_1", "solver_2"),
        ("solver_1", "solver_3"),
        ("solver_2", "solver_1"),
        ("solver_2", "solver_3"),
        ("solver_3", "solver_1"),
        ("solver_3", "solver_2"),
    ]
    for reviewer, target in pairs:
        reviewer_profile = SOLVER_PROFILES.get(reviewer, default_profile)
        review = stage2_peer_review(
            problem,
            reviewer,
            role_to_agent[reviewer],
            target,
            stage1[target],
            solver_models[reviewer],
            reviewer_profile["review_temp"],
            reviewer_profile["style"],
        )
        stage2.append(review)

    stage3 = {}
    for solver_label in ["solver_1", "solver_2", "solver_3"]:
        profile = SOLVER_PROFILES.get(solver_label, default_profile)
        reviews = [r for r in stage2 if r.get("solution_id", "").lower() == solver_label]
        stage3[solver_label] = stage3_refine(
            problem,
            solver_label,
            role_to_agent[solver_label],
            stage1[solver_label],
            reviews,
            solver_models[solver_label],
            profile["refine_temp"],
            profile["style"],
        )

    stage4 = stage4_judge(problem, role_to_agent["judge"], stage1, stage2, stage3, MODEL_JUDGE, 0.2)
    winner = stage4.get("winner", "solver_1").lower()
    final_answer = stage3.get(winner, {}).get("refined_answer", "")

    return {
        "problem": {
            "id": problem.id,
            "category": problem.category,
            "question": problem.question,
            "ground_truth": problem.ground_truth,
            "answer_type": problem.answer_type,
        },
        "stage0": stage0,
        "stage0_5": role_map,
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4,
        "final_answer": final_answer,
        "winner": winner,
    }


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(records)
    if total == 0:
        return {}
    correct_final = 0
    judge_correct = 0
    judge_cases = 0
    consensus = 0
    improvement = 0
    single_baseline = 0
    voting_baseline = 0

    for rec in records:
        prob = rec["problem"]
        expected = prob["ground_truth"]
        answer_type = prob["answer_type"]
        final_answer = rec.get("final_answer", "")
        if is_correct(final_answer, expected, answer_type):
            correct_final += 1

        refined_answers = [
            rec.get("stage3", {}).get("solver_1", {}).get("refined_answer", ""),
            rec.get("stage3", {}).get("solver_2", {}).get("refined_answer", ""),
            rec.get("stage3", {}).get("solver_3", {}).get("refined_answer", ""),
        ]
        normed = [_normalize_text(a) for a in refined_answers if a is not None]
        disagreement = len(set(normed)) > 1 if normed else False
        if len(set(normed)) == 1 and normed:
            consensus += 1
        winner = rec.get("winner", "solver_1")
        winner_refined = rec.get("stage3", {}).get(winner, {}).get("refined_answer", "")
        winner_original = rec.get("stage1", {}).get(winner, {}).get("final_answer", "")
        if disagreement:
            judge_cases += 1
            if is_correct(winner_refined, expected, answer_type):
                judge_correct += 1
        if is_correct(winner_refined, expected, answer_type) and not is_correct(winner_original, expected, answer_type):
            improvement += 1

        solver1_original = rec.get("stage1", {}).get("solver_1", {}).get("final_answer", "")
        if is_correct(solver1_original, expected, answer_type):
            single_baseline += 1

        vote_counts: Dict[str, int] = {}
        for ans in refined_answers:
            key = _normalize_text(ans)
            vote_counts[key] = vote_counts.get(key, 0) + 1
        if vote_counts:
            voted = max(vote_counts.items(), key=lambda kv: kv[1])[0]
            if is_correct(voted, expected, answer_type):
                voting_baseline += 1

    return {
        "overall_accuracy": correct_final / total,
        "improvement_rate": improvement / total,
        "consensus_rate": consensus / total,
        "judge_accuracy": (judge_correct / judge_cases) if judge_cases else 0.0,
        "single_llm_baseline": single_baseline / total,
        "simple_voting_baseline": voting_baseline / total,
    }


def plot_metrics(metrics: Dict[str, float]) -> None:
    labels = [
        "Debate System Accuracy",
        "Single LLM Accuracy",
        "Voting Accuracy",
        "Judge Accuracy (disagreement only)",
        "Improvement Rate",
        "Consensus Rate",
    ]
    values = [
        metrics.get("overall_accuracy", 0.0),
        metrics.get("single_llm_baseline", 0.0),
        metrics.get("simple_voting_baseline", 0.0),
        metrics.get("judge_accuracy", 0.0),
        metrics.get("improvement_rate", 0.0),
        metrics.get("consensus_rate", 0.0),
    ]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color="steelblue")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics.png"), dpi=150)
    plt.close()


def main() -> None:
    dataset = build_dataset()
    if not dataset:
        print("No problems found.")
        return
    # if len(dataset) != 25:
    #     raise ValueError(f"Dataset must contain exactly 25 problems, found {len(dataset)}.")

    first_record = run_pipeline(dataset[0])
    print(f"First problem final answer: {first_record.get('final_answer')}")

    all_records = [first_record]
    for problem in dataset[1:]:
        rec = run_pipeline(problem)
        all_records.append(rec)
        print(f"{problem.id} final answer: {rec.get('final_answer')}")

    with open(os.path.join(RESULTS_DIR, "outputs.json"), "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    metrics = compute_metrics(all_records)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plot_metrics(metrics)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
