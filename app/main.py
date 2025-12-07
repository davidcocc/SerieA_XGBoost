from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .predictor import PredictorService


app = FastAPI(title="Serie A Match Predictor") # web app title
predictor = PredictorService() # predictor service

BADGES_DIR = Path(__file__).resolve().parents[1] / "badges"
if BADGES_DIR.exists():
    app.mount("/badges", StaticFiles(directory=str(BADGES_DIR)), name="badges")

TEAM_BADGES: Dict[str, str] = {}

placeholder_file = BADGES_DIR / "placeholder.png"
PLACEHOLDER_BADGE = (
    f"/badges/{placeholder_file.name}" if placeholder_file.exists() else ""
)


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _build_badge_map(teams: List[str]) -> Dict[str, str]:
    badge_map: Dict[str, str] = {}
    if not BADGES_DIR.exists():
        return badge_map

    badge_files = list(BADGES_DIR.glob("*.png"))
    for team in teams:
        norm_team = _normalize(team)
        match_name = None

        for file in badge_files:
            if _normalize(file.stem) == norm_team:
                match_name = file.name
                break

        if not match_name:
            for file in badge_files:
                norm_file = _normalize(file.stem)
                if norm_team in norm_file or norm_file in norm_team:
                    match_name = file.name
                    break

        if match_name:
            badge_map[team] = f"/badges/{match_name}"
    return badge_map


class MatchRequest(BaseModel):
    team: str
    opponent: str
    venue: str = Field(pattern="^(Home|Away)$")
    formation: str
    opp_formation: str = Field(alias="oppFormation")

    class Config:
        allow_population_by_field_name = True


def _badge_for(team: str) -> str:
    return TEAM_BADGES.get(team, PLACEHOLDER_BADGE)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    teams = predictor.list_teams()
    formations = predictor.list_formations()
    global TEAM_BADGES
    TEAM_BADGES = _build_badge_map(teams)
    badge_map = {team: _badge_for(team) for team in teams}
    badge_map_json = json.dumps(badge_map)

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Serie A Match Predictor</title>
        <link rel="preconnect" href="https://fonts.gstatic.com" />
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" />
        <style>
            body {{
                font-family: 'Roboto', Arial, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(180deg, #07131b 0%, #0c3326 55%, #0f442f 100%);
                color: #f9fafb;
                min-height: 100vh;
            }}
            header {{
                background: rgba(4, 18, 26, 0.92);
                padding: 3rem 2rem;
                text-align: center;
                border-bottom: 4px solid rgba(29, 211, 102, 0.65);
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
            }}
            main {{
                padding: 2.5rem 2rem 4rem;
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1 {{
                font-size: 2.6rem;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                margin: 0;
            }}
            h1 span {{
                color: #1dd366;
            }}
            .subtitle {{
                margin-top: 0.75rem;
                font-size: 1rem;
                color: #cdd5d9;
            }}
            .card {{
                background: rgba(255, 255, 255, 0.06);
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 20px 45px rgba(0, 0, 0, 0.35);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }}
            .teams {{
                display: flex;
                gap: 1.5rem;
                justify-content: space-between;
                flex-wrap: wrap;
                margin-bottom: 2rem;
            }}
            .team-panel {{
                flex: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 1rem;
                padding: 1.5rem 1rem;
                border-radius: 10px;
                background: rgba(9, 29, 36, 0.8);
                border: 1px solid rgba(29, 211, 102, 0.25);
                position: relative;
            }}
            .team-panel img {{
                width: 110px;
                height: 110px;
                object-fit: contain;
                filter: drop-shadow(0 6px 18px rgba(0, 0, 0, 0.6));
            }}
            label {{
                font-weight: 600;
                color: #a0f2c9;
            }}
            select {{
                width: 100%;
                padding: 0.75rem;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                font-size: 1rem;
                background: rgba(6, 19, 24, 0.9);
                color: #f9fafb;
                box-shadow: inset 0 0 14px rgba(0, 0, 0, 0.25);
            }}
            .venue {{
                margin: 2rem 0;
                text-align: center;
            }}
            button {{
                background: linear-gradient(120deg, #1dd366 0%, #0aa852 100%);
                color: #062212;
                border: none;
                padding: 0.9rem 2.8rem;
                border-radius: 999px;
                font-size: 1rem;
                font-weight: 700;
                cursor: pointer;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                box-shadow: 0 14px 32px rgba(29, 211, 102, 0.35);
            }}
            button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 18px 38px rgba(29, 211, 102, 0.45);
            }}
            .results {{
                margin-top: 2rem;
                text-align: center;
            }}
            .results h2 {{
                margin-bottom: 0.5rem;
                color: #9be7c4;
            }}
            .result-card {{
                display: inline-block;
                background: rgba(6, 24, 18, 0.92);
                color: #22f38a;
                padding: 1.25rem 2.8rem;
                border-radius: 10px;
                font-size: 2rem;
                min-width: 230px;
                font-weight: 700;
                border: 2px solid rgba(34, 243, 138, 0.4);
                letter-spacing: 0.1em;
            }}
            .prediction-details {{
                margin-top: 1rem;
                color: #d4dde0;
            }}
            .form-dots {{
                display: flex;
                gap: 0.45rem;
                margin-top: 0.4rem;
            }}
            .form-dot {{
                width: 16px;
                height: 16px;
                border-radius: 50%;
                border: 1px solid rgba(255, 255, 255, 0.35);
                background: rgba(255, 255, 255, 0.15);
                box-shadow: 0 0 6px rgba(0, 0, 0, 0.3);
            }}
            .form-dot.win {{ background: #1dd366; }}
            .form-dot.draw {{ background: #facc15; }}
            .form-dot.loss {{ background: #f87171; }}
            .tabs {{
                display: flex;
                gap: 1rem;
                margin-bottom: 1.5rem;
            }}
            .tab {{
                flex: 1;
                padding: 0.95rem;
                text-align: center;
                border-radius: 12px;
                background: rgba(9, 30, 36, 0.7);
                color: #a0f2c9;
                font-weight: 600;
                cursor: pointer;
                border: 1px solid rgba(29, 211, 102, 0.2);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            .tab.active {{
                background: rgba(29, 211, 102, 0.25);
                color: #f9fafb;
                border-color: rgba(29, 211, 102, 0.6);
                box-shadow: 0 12px 26px rgba(29, 211, 102, 0.25);
                transform: translateY(-2px);
            }}
            .tab-content {{
                display: none;
            }}
            .tab-content.active {{
                display: block;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
                gap: 1.5rem;
                margin-top: 2rem;
            }}
            .stats-card {{
                background: rgba(6, 19, 24, 0.85);
                padding: 1.5rem;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.12);
                box-shadow: inset 0 0 35px rgba(0, 0, 0, 0.25);
            }}
            .stats-card h3 {{
                margin: 0 0 1rem 0;
                color: #9be7c4;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }}
            .stats-table {{
                width: 100%;
                border-collapse: collapse;
                color: #d1d5db;
            }}
            .stats-table th, .stats-table td {{
                padding: 0.55rem 0.6rem;
                text-align: left;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                font-size: 0.92rem;
            }}
            .stats-table th {{
                text-transform: uppercase;
                font-size: 0.75rem;
                letter-spacing: 0.08em;
                color: #9be7c4;
            }}
            .stats-table tr:last-child td {{
                border-bottom: none;
            }}
            .chart-tooltip {{
                position: absolute;
                pointer-events: none;
                background: rgba(6, 19, 24, 0.95);
                border: 1px solid rgba(29, 211, 102, 0.4);
                border-radius: 10px;
                padding: 0.75rem 1rem;
                color: #f9fafb;
                box-shadow: 0 12px 25px rgba(0, 0, 0, 0.35);
                transform: translate(-50%, -110%);
                transition: all 0.1s ease;
                min-width: 180px;
                z-index: 30;
            }}
            .chart-tooltip h4 {{
                margin: 0 0 0.4rem 0;
                font-size: 1rem;
                color: #9be7c4;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }}
            .chart-tooltip img {{
                width: 48px;
                height: 48px;
                object-fit: contain;
                display: block;
                margin-bottom: 0.4rem;
                filter: drop-shadow(0 4px 10px rgba(0, 0, 0, 0.45));
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.6/dist/chart.umd.min.js"></script>
    </head>
    <body>
        <header>
            <h1><span>Serie A</span> Match Center</h1>
            <p class="subtitle">Pre-match insights, tactical exploration, and model-driven projections strictly for learning purposes. <strong>Do NOT use this tool for betting.</strong></p>
        </header>
        <main>
            <div class="tabs">
                <div class="tab active" data-tab="predictTab">Matchboard</div>
                <div class="tab" data-tab="statsTab">Analytics</div>
            </div>
            <div class="card">
                <div id="predictTab" class="tab-content active">
                    <div class="teams">
                        <div class="team-panel">
                            <img id="homeBadge" src="{PLACEHOLDER_BADGE}" alt="Team badge" />
                            <label for="teamSelect">Squad</label>
                            <select id="teamSelect">
                                {"".join(f'<option value="{team}">{team}</option>' for team in teams)}
                            </select>
                            <label for="formationSelect">Formation</label>
                            <select id="formationSelect">
                                {"".join(f'<option value="{formation}">{formation}</option>' for formation in formations)}
                            </select>
                            <div class="form-dots" id="teamFormDots"></div>
                        </div>
                        <div class="team-panel">
                            <img id="awayBadge" src="{PLACEHOLDER_BADGE}" alt="Opponent badge" />
                            <label for="opponentSelect">Opponent</label>
                            <select id="opponentSelect">
                                {"".join(f'<option value="{team}">{team}</option>' for team in teams)}
                            </select>
                            <label for="oppFormationSelect">Opponent Formation</label>
                            <select id="oppFormationSelect">
                                {"".join(f'<option value="{formation}">{formation}</option>' for formation in formations)}
                            </select>
                            <div class="form-dots" id="opponentFormDots"></div>
                        </div>
                    </div>
                    <div class="venue">
                        <label>Venue</label>
                        <select id="venueSelect">
                            <option value="Home">Home</option>
                            <option value="Away">Away</option>
                        </select>
                    </div>
                    <div style="text-align: center;">
                        <button id="predictBtn">Kick Off</button>
                    </div>
                    <div class="results" id="results" style="display: none;">
                        <h2>Projected Scoreline</h2>
                        <div class="result-card" id="scoreline">0 - 0</div>
                        <div class="prediction-details" id="details"></div>
                        <div class="stats-card" id="headToHeadSection" style="margin-top: 1.5rem; display: none;">
                            <h3>Head-to-Head (Last 5)</h3>
                            <div id="headToHeadBar" style="display: flex; gap: 0.4rem;"></div>
                        </div>
                    </div>
                </div>
                <div id="statsTab" class="tab-content">
                    <div class="stats-grid">
                        <div class="stats-card">
                            <h3>xG Created vs Goals Scored</h3>
                            <canvas id="gfGaChart" height="320"></canvas>
                        </div>
                        <div class="stats-card">
                            <h3>Goal Difference Table</h3>
                            <table class="stats-table">
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Club</th>
                                        <th>Diff</th>
                                        <th>Win%</th>
                                    </tr>
                                </thead>
                                <tbody id="goalDiffTable">
                                    <tr><td colspan="4">Loading statistics...</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </main>
        <script>
            const badges = {badge_map_json};
            const formColors = {{
                "W": "win",
                "D": "draw",
                "L": "loss",
            }};

            const labelPlugin = {{
                id: "teamLabels",
                afterDatasetsDraw(chart) {{
                    const {{ ctx, data }} = chart;
                    const meta = chart.getDatasetMeta(0);
                    meta.data.forEach((element, index) => {{
                        const {{
                            x,
                            y
                        }} = element.tooltipPosition();
                        const team = data.datasets[0].data[index].team;
                        ctx.save();
                        ctx.font = "700 11px Roboto";
                        ctx.fillStyle = "#9be7c4";
                        ctx.textAlign = "center";
                        ctx.fillText(team.slice(0, 3).toUpperCase(), x, y - 16);
                        ctx.restore();
                    }});
                }}
            }};
            Chart.register(labelPlugin);

            function externalTooltipHandler(context) {{
                const {{ chart, tooltip }} = context;
                let tooltipEl = document.getElementById("chartjs-tooltip");
                if (!tooltipEl) {{
                    tooltipEl = document.createElement("div");
                    tooltipEl.id = "chartjs-tooltip";
                    tooltipEl.classList.add("chart-tooltip");
                    tooltipEl.style.opacity = 0;
                    document.body.appendChild(tooltipEl);
                }}

                if (tooltip.opacity === 0) {{
                    tooltipEl.style.opacity = 0;
                    return;
                }}

                const data = tooltip.dataPoints[0].raw;
                tooltipEl.innerHTML = `
                    <img src="${{data.badge}}" alt="${{data.team}} badge" />
                    <h4>${{data.team}}</h4>
                    <div>Average xG: <strong>${{data.x.toFixed(2)}}</strong></div>
                    <div>Goals Scored: <strong>${{data.y.toFixed(2)}}</strong></div>
                    <div>Win Rate: <strong>${{data.winRate.toFixed(1)}}%</strong></div>
                `;

                const {{ offsetLeft: positionX, offsetTop: positionY }} = chart.canvas;
                tooltipEl.style.opacity = 1;
                tooltipEl.style.left = positionX + tooltip.caretX + "px";
                tooltipEl.style.top = positionY + tooltip.caretY + "px";
            }}

            async function loadRecentForm(team, elementId) {{
                const response = await fetch(`/recent/${{encodeURIComponent(team)}}`);
                if (!response.ok) {{
                    return;
                }}
                const data = await response.json();
                const dotsContainer = document.getElementById(elementId);
                dotsContainer.innerHTML = "";
                data.results.forEach(result => {{
                    const dot = document.createElement("div");
                    dot.classList.add("form-dot", formColors[result] || "draw");
                    dot.title = result;
                    dotsContainer.appendChild(dot);
                }});
            }}

            async function updateSelections() {{
                const team = document.getElementById("teamSelect").value;
                const opponent = document.getElementById("opponentSelect").value;
                document.getElementById("homeBadge").src = badges[team] || "{PLACEHOLDER_BADGE}";
                document.getElementById("awayBadge").src = badges[opponent] || "{PLACEHOLDER_BADGE}";
                await Promise.all([
                    loadRecentForm(team, "teamFormDots"),
                    loadRecentForm(opponent, "opponentFormDots")
                ]);
            }}

            document.getElementById("teamSelect").addEventListener("change", updateSelections);
            document.getElementById("opponentSelect").addEventListener("change", updateSelections);
            updateSelections();

            async function predict() {{
                const payload = {{
                    team: document.getElementById("teamSelect").value,
                    opponent: document.getElementById("opponentSelect").value,
                    venue: document.getElementById("venueSelect").value,
                    formation: document.getElementById("formationSelect").value,
                    oppFormation: document.getElementById("oppFormationSelect").value
                }};

                if (payload.team === payload.opponent) {{
                    alert("Please select two different teams.");
                    return;
                }}

                const response = await fetch("/predict", {{
                    method: "POST",
                    headers: {{
                        "Content-Type": "application/json"
                    }},
                    body: JSON.stringify(payload)
                }});

                if (!response.ok) {{
                    const error = await response.json();
                    alert(error.detail || "Error during prediction. Match suspended.");
                    return;
                }}

                const data = await response.json();
                document.getElementById("results").style.display = "block";
                document.getElementById("scoreline").textContent = `${{data.pred_goals_for_round}} - ${{data.pred_goals_against_round}}`;
                document.getElementById("details").textContent = `Outcome: ${{formatResult(data.result)}} | GF: ${{data.pred_goals_for.toFixed(2)}}, GA: ${{data.pred_goals_against.toFixed(2)}}`;
                await Promise.all([
                    loadHeadToHead(payload.team, payload.opponent),
                    updateSelections()
                ]);
            }}

            document.getElementById("predictBtn").addEventListener("click", predict);
            function formatResult(result) {{
                if (result === "W") return "Win";
                if (result === "L") return "Loss";
                return "Draw";
            }}

            async function loadHeadToHead(team, opponent) {{
                const response = await fetch(`/head-to-head?team=${{encodeURIComponent(team)}}&opponent=${{encodeURIComponent(opponent)}}`);
                const section = document.getElementById("headToHeadSection");
                const bar = document.getElementById("headToHeadBar");
                if (!response.ok) {{
                    section.style.display = "none";
                    return;
                }}
                const records = await response.json();
                if (!records.length) {{
                    section.style.display = "none";
                    return;
                }}
                section.style.display = "block";
                bar.innerHTML = "";
                records.forEach(record => {{
                    const block = document.createElement("div");
                    block.classList.add("form-dot");
                    const normalized = record.result === "W"
                        ? (record.team === team ? "win" : "loss")
                        : record.result === "L"
                            ? (record.team === team ? "loss" : "win")
                            : "draw";
                    block.classList.add(normalized);
                    block.title = `${{record.date}} - ${{record.team}} ${{record.gf}}:${{record.ga}} ${{record.opponent}}`;
                    bar.appendChild(block);
                }});
            }}

            const tabs = document.querySelectorAll(".tab");
            const tabContents = document.querySelectorAll(".tab-content");
            tabs.forEach(tab => {{
                tab.addEventListener("click", () => {{
                    tabs.forEach(t => t.classList.remove("active"));
                    tabContents.forEach(c => c.classList.remove("active"));
                    tab.classList.add("active");
                    document.getElementById(tab.dataset.tab).classList.add("active");
                    if (tab.dataset.tab === "statsTab") {{
                        loadStats();
                    }}
                }});
            }});

            let statsLoaded = false;

            async function loadStats() {{
                if (statsLoaded) {{
                    return;
                }}
                const response = await fetch("/stats");
                if (!response.ok) {{
                    console.error("Unable to retrieve statistics from the server.");
                    return;
                }}
                const stats = await response.json();
                renderStats(stats);
                statsLoaded = true;
            }}

            function renderStats(stats) {{
                const dataPoints = stats.map(item => {{
                    const img = new Image();
                    img.src = badges[item.team] || "{PLACEHOLDER_BADGE}";
                    return {{
                        x: item.avg_xg,
                        y: item.avg_gf,
                        team: item.team,
                        goalDiff: item.goal_diff,
                        badge: img.src,
                        winRate: item.win_rate,
                    }};
                }});

                const ctx = document.getElementById("gfGaChart").getContext("2d");
                new Chart(ctx, {{
                    type: "scatter",
                    data: {{
                        datasets: [{{
                            data: dataPoints,
                            pointRadius: 6,
                            pointHoverRadius: 10,
                            pointBackgroundColor: "rgba(29, 211, 102, 0.7)",
                            pointBorderColor: "rgba(29, 211, 102, 1)",
                            pointBorderWidth: 1.5,
                        }}]
                    }},
                    options: {{
                        plugins: {{
                            legend: {{
                                display: false
                            }},
                            tooltip: {{
                                enabled: false,
                                external: externalTooltipHandler
                            }},
                        }},
                        scales: {{
                            x: {{
                                title: {{
                                    display: true,
                                    text: "Average xG produced",
                                    color: "#9be7c4"
                                }},
                                ticks: {{
                                    color: "#d1d5db"
                                }},
                                grid: {{
                                    color: "rgba(255, 255, 255, 0.08)"
                                }}
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: "Average goals scored",
                                    color: "#9be7c4"
                                }},
                                ticks: {{
                                    color: "#d1d5db"
                                }},
                                grid: {{
                                    color: "rgba(255, 255, 255, 0.08)"
                                }}
                            }}
                        }}
                    }}
                }});

                const sorted = [...stats].sort((a, b) => b.goal_diff - a.goal_diff);
                const tbody = document.getElementById("goalDiffTable");
                tbody.innerHTML = "";
                sorted.forEach((item, index) => {{
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td>${{index + 1}}</td>
                        <td>${{item.team}}</td>
                        <td>${{item.goal_diff}}</td>
                        <td>${{item.win_rate.toFixed(1)}}%</td>
                    `;
                    tbody.appendChild(tr);
                }});
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/predict")
def predict_match(request: MatchRequest) -> JSONResponse:
    if request.team == request.opponent:
        raise HTTPException(status_code=400, detail="Teams must be different. You can't play against yourself!")

    prediction = predictor.predict(
        team=request.team,
        opponent=request.opponent,
        venue=request.venue,
        formation=request.formation,
        opp_formation=request.opp_formation,
    )
    return JSONResponse(prediction)


@app.get("/stats")
def get_stats() -> JSONResponse:
    stats = predictor.team_summary()
    return JSONResponse(stats)


@app.get("/recent/{team}")
def get_recent(team: str) -> JSONResponse:
    results = predictor.recent_results(team)
    return JSONResponse({"team": team, "results": results})


@app.get("/head-to-head")
def get_head_to_head(team: str, opponent: str) -> JSONResponse:
    records = predictor.head_to_head(team, opponent)
    return JSONResponse(records)

