"""Fact-grounded reasoning strings.

Every claim is pulled from the candidate's actual profile (title, years,
named evidence terms found in THEIR career descriptions, their signal
values). Sentence structure varies deterministically by candidate_id, and
each entry names the candidate's strongest evidence plus their most
material concern — per the Stage 4 rubric (specific facts, JD connection,
honest concerns, no hallucination, variation, rank consistency).
"""


def _evidence_phrase(facts: dict) -> str | None:
    hits = facts["evidence_hits"]
    core = hits.get("retrieval", []) + hits.get("ranking", [])
    if core:
        # prefer distinctive, multi-word evidence over generic terms
        ordered = sorted(set(core), key=lambda t: (-len(t), t))
        named = ", ".join(ordered[:3])
        return f"career history shows hands-on work with {named}"
    nlp = hits.get("nlp_llm", [])
    if nlp:
        return f"NLP/LLM exposure ({', '.join(sorted(set(nlp))[:2])}) in role descriptions"
    return None


def _top_skills_phrase(facts: dict) -> str | None:
    strong = sorted(
        (s for s in facts["skills"] if s[2] >= 0.5),
        key=lambda s: s[1] * s[2],
        reverse=True,
    )[:3]
    if not strong:
        return None
    return "corroborated skills in " + ", ".join(s[0] for s in strong)


def _concern_phrase(c: dict, facts: dict, components: dict) -> str | None:
    beh = facts["behavioral"]
    concerns = []
    if beh["notice_days"] > 60:
        concerns.append(f"{beh['notice_days']}-day notice period")
    if beh["months_inactive"] > 3:
        concerns.append(f"last active ~{beh['months_inactive']:.0f} months ago")
    if beh["response_rate"] < 0.4:
        concerns.append(f"recruiter response rate of {beh['response_rate']:.0%}")
    if components["location"] < 0.5:
        concerns.append(f"based in {c['profile']['location']}")
    yoe = c["profile"]["years_of_experience"]
    if yoe < 5:
        concerns.append(f"only {yoe:.1f} yrs experience vs the 5-9 band")
    elif yoe > 9:
        concerns.append(f"{yoe:.1f} yrs experience, above the 5-9 band")
    if facts["tenure"]["hopper"]:
        concerns.append(f"short average tenure (~{facts['tenure']['avg_tenure']:.0f} months)")
    if not concerns:
        return None
    return concerns[0]


def build_reasoning(c: dict, sc, rank: int) -> str:
    p = c["profile"]
    facts, components = sc.facts, sc.components
    yoe = p["years_of_experience"]
    title = p["current_title"]
    company = p["current_company"]
    variant = int(c["candidate_id"][5:]) % 4

    ev = _evidence_phrase(facts)
    sk = _top_skills_phrase(facts)
    concern = _concern_phrase(c, facts, components)
    beh = facts["behavioral"]

    engaged = (
        f"responsive on platform ({beh['response_rate']:.0%} response rate)"
        if beh["response_rate"] >= 0.7 and beh["months_inactive"] <= 1
        else None
    )

    bits = []
    if variant == 0:
        bits.append(f"{title} at {company} with {yoe:.1f} yrs")
    elif variant == 1:
        bits.append(f"{yoe:.1f} yrs of experience, currently {title} at {company}")
    elif variant == 2:
        bits.append(f"Currently {title} ({company}), {yoe:.1f} yrs total")
    else:
        bits.append(f"{title} with {yoe:.1f} yrs, now at {company}")

    if ev:
        bits.append(ev)
    if sk and (rank <= 60 or not ev):
        bits.append(sk)
    if engaged and rank <= 50:
        bits.append(engaged)

    sentence1 = "; ".join(bits) + "."

    if concern:
        if rank <= 25:
            sentence2 = f" Main caveat: {concern}."
        elif rank <= 70:
            sentence2 = f" Concern: {concern}."
        else:
            sentence2 = f" Ranked lower due to {concern}."
    else:
        sentence2 = ""
        if rank > 70 and components["evidence"] < 0.4:
            sentence2 = " Adjacent rather than direct retrieval/ranking experience keeps them in the lower tier."

    text = (sentence1 + sentence2).replace('"', "'")
    return text
