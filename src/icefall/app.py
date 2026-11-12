"""Gradio front end. All logic lives in ``score.score_climber`` so it can be unit tested."""

from __future__ import annotations

from . import score


def explain(result: dict) -> str:
    lines = [
        f"### {result['peak']} ({result['height_m']:,} m)",
        f"Estimated chance of reaching the summit: **{result['p_summit'] * 100:.0f}%** ({result['band']}).",
        f"Recent rate for all non-hired climbers on this peak: {result['peak_recent_rate'] * 100:.0f}%.",
    ]
    if result["p_oxygen"] > 0:
        lines.append(f"Chance this climber ends up using bottled oxygen: {result['p_oxygen'] * 100:.0f}%.")
    lines.append(
        f"Chance of dying on this trip: about {result['p_death'] * 100:.1f}%. "
        "This estimate is weak (see the README), so treat it as a rough base rate."
    )
    lines.append("\nWhat moves the estimate:")
    lines += [f"- {r}" for r in result["reasons"]]
    return "\n".join(lines)


def predict(bundle, peak_name, season, age, sex, nationality, role, team_size, hired_staff, solo, agency) -> str:
    row = bundle.peaks[bundle.peaks["peak_name"] == peak_name]
    if row.empty:
        return "Pick a peak from the list."
    res = score.score_climber(
        bundle,
        row.iloc[0]["peak_id"],
        season=season,
        age=age if age and age > 0 else None,
        sex=sex,
        citizenship=nationality,
        role=role,
        team_size=team_size,
        hired_staff=hired_staff,
        solo=int(bool(solo)),
        has_agency=int(bool(agency)),
    )
    return explain(res)


def build_app(bundle):
    import gradio as gr

    peaks = bundle.peaks["peak_name"].tolist()
    with gr.Blocks(title="icefall") as demo:
        gr.Markdown(
            "# icefall\nBooking-time odds for a paying climber in the Nepal Himalaya, "
            "from 1950 to 2019 expedition records. Not safety advice."
        )
        with gr.Row():
            peak = gr.Dropdown(peaks, value="Everest" if "Everest" in peaks else peaks[0], label="Peak")
            season = gr.Dropdown(score.SEASONS, value="Spring", label="Season")
            role = gr.Dropdown(score.ROLES, value="climber", label="Role")
        with gr.Row():
            age = gr.Number(value=40, label="Age (0 if unknown)")
            sex = gr.Dropdown(["M", "F"], value="M", label="Sex")
            nation = gr.Dropdown(bundle.nations, value=bundle.nations[0], label="Citizenship")
        with gr.Row():
            team = gr.Slider(1, 40, value=10, step=1, label="Team size")
            staff = gr.Slider(0, 40, value=8, step=1, label="Hired staff above base camp")
            solo = gr.Checkbox(False, label="Solo attempt")
            agency = gr.Checkbox(True, label="Booked through an agency")
        out = gr.Markdown()
        btn = gr.Button("Estimate")
        btn.click(
            lambda *a: predict(bundle, *a),
            [peak, season, age, sex, nation, role, team, staff, solo, agency],
            out,
        )
    return demo


def main():
    build_app(score.load_bundle()).launch()


if __name__ == "__main__":
    main()
