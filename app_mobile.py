import json
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

# ---------------------------------------------------------
# PAGE CONFIGURATION & RESPONSIVE STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="EuroLeague Fantasy Dominator Pro 2026/27",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stMetric {
        background-color: #161b22;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    .stButton>button {
        width: 100%;
        background-color: #f78166;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 10px;
    }
    .stButton>button:hover { background-color: #e05d00; color: white; }
    @media (max-width: 768px) {
        .stMetric { padding: 8px; font-size: 12px; }
        .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stTabs [data-baseweb="tab"] { font-size: 11px; padding: 4px 8px; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TRANSLATION DICTIONARY (LT / EN)
# ---------------------------------------------------------
TRANSLATIONS = {
    "LT": {
        "title": "🏀 EuroLeague Fantasy Dominator Pro",
        "subtitle": "Algoritminis sudėčių optimizatorius • Matchup Variklis",
        "lang_select": "🌍 Kalba / Language",
        "source_select": "⚙️ Duomenų šaltinis",
        "src_sample": "Pavyzdiniai duomenys (Demo)",
        "src_json": "Įklijuoti BasketNews API JSON",
        "tab_overview": "🏠 Apžvalga",
        "tab_injuries": "🚑 Traumos ir Rotacijos",
        "tab_schedule": "📅 Tvarkaraštis ir Dienos",
        "tab_optimizer": "🤖 Optimizatorius ir Kapitono Taktika",
        "tab_roster": "📝 Mano Sudėtis",
        "tab_database": "🔍 Žaidėjų Bazė",
        "total_players": "Viso Žaidėjų",
        "avg_pir": "Vidutinis PIR",
        "top_val": "Geriausia Vertė",
        "budget": "Biudžetas (€M)",
        "generate_opt": "⚡ Generuoti Optimalią Sudėtį ir Kapitonus",
        "opt_success": "✅ Optimali sudėtis ir Kapitonų strategija sėkmingai parinkta!",
        "opt_fail": "Nepavyko rasti sudėties su šiuo biudžetu.",
        "proj_pir": "Prognozuojamas PIR",
        "c_day1": "👑 1-osios Dienos Kapitonas",
        "c_day2": "🔄 2-osios Dienos Kapitonas",
        "c_day3": "🚀 3-iosios Dienos Kapitonas",
        "day_1": "1 Diena",
        "day_2": "2 Diena",
        "day_3": "3 Diena",
        "injury_sim": "🚑 Traumuotų Žaidėjų Poveikio Simuliatorius",
        "out_players": "Pažymėkite NEŽAIDŽIANČIUS krepšininkus:",
        "col_name": "Žaidėjas",
        "col_team": "Komanda",
        "col_opp": "Varžovas",
        "col_pos": "Pozicija",
        "col_price": "Kaina (€M)",
        "col_pir": "Vid. PIR",
        "col_proj_pir": "Prog. PIR",
        "col_val": "Vertė",
        "col_day": "Diena",
        "col_status": "Būsena"
    },
    "EN": {
        "title": "🏀 EuroLeague Fantasy Dominator Pro",
        "subtitle": "Algorithmic Roster Optimizer • Matchup Difficulty Engine",
        "lang_select": "🌍 Kalba / Language",
        "source_select": "⚙️ Data Source",
        "src_sample": "Sample Data (Demo)",
        "src_json": "Paste BasketNews API JSON",
        "tab_overview": "🏠 Overview",
        "tab_injuries": "🚑 Injuries & Rotations",
        "tab_schedule": "📅 Schedule & Game Days",
        "tab_optimizer": "🤖 Optimizer & Captain Strategy",
        "tab_roster": "📝 My Roster",
        "tab_database": "🔍 Player Database",
        "total_players": "Total Players",
        "avg_pir": "Average PIR",
        "top_val": "Top Value Player",
        "budget": "Budget (€M)",
        "generate_opt": "⚡ Generate Optimal Roster & Captains",
        "opt_success": "✅ Optimal lineup and Multi-Day Captains selected successfully!",
        "opt_fail": "Could not find valid lineup within budget.",
        "proj_pir": "Projected PIR",
        "c_day1": "👑 Day 1 Captain",
        "c_day2": "🔄 Day 2 Captain",
        "c_day3": "🚀 Day 3 Captain",
        "day_1": "Day 1",
        "day_2": "Day 2",
        "day_3": "Day 3",
        "injury_sim": "🚑 Injury Impact & Usage Shift Simulator",
        "out_players": "Select OUT / Injured Players:",
        "col_name": "Player",
        "col_team": "Team",
        "col_opp": "Opponent",
        "col_pos": "Pos",
        "col_price": "Price (€M)",
        "col_pir": "Avg PIR",
        "col_proj_pir": "Proj PIR",
        "col_val": "Value Score",
        "col_day": "Game Day",
        "col_status": "Status"
    }
}

st.sidebar.title("🌐 Settings")
lang = st.sidebar.selectbox("Language / Kalba", ["LT", "EN"])
t = TRANSLATIONS[lang]

# ---------------------------------------------------------
# DEFENSIVE RATINGS & MATCHUP ENGINE
# ---------------------------------------------------------
# Varžovų gynybos reitingas (1 = silpniausia gynyba; 18 = elitinė gynyba)
OPPONENT_DEFENSE_RANK = {
    "ALBA": 1, "ASVEL": 3, "Paris": 5, "Virtus": 6, "Baskonia": 8, 
    "Valencia": 9, "Efes": 10, "Partizan": 11, "Maccabi": 12, "Zalgiris": 13, 
    "Olympiakos": 15, "Fenerbahce": 16, "PAO": 17, "Real Madrid": 18
}

def calculate_advanced_proj(base_pir, opp_team):
    """ Skaičiuoja Proj PIR atsižvelgiant TIK į varžovo gynybos stiprumą """
    def_rank = OPPONENT_DEFENSE_RANK.get(opp_team, 10)
    
    # Matchup koeficientas nuo +15% (vs ALBA) iki -15% (vs Real Madrid)
    matchup_modifier = 1.15 - ((def_rank - 1) * 0.017)
    
    proj = base_pir * matchup_modifier
    return round(max(0.0, proj), 1)

# ---------------------------------------------------------
# DATA PARSER
# ---------------------------------------------------------
def parse_basketnews_json(raw_json):
    records = raw_json.get("data", {}).get("playersSearchRecordsFromClient", {}).get("records", [])
    players, schedule_dict, dates_found = [], {}, []
    pos_mapping = {"guard": "G", "forward": "F", "center": "C"}

    for item in records:
        team_info = item.get("team", {}).get("team", {}) or {}
        for g in team_info.get("games", []):
            game_date = g.get("originalGameAt", "")[:10]
            if game_date and game_date not in dates_found:
                dates_found.append(game_date)
    
    dates_found = sorted(dates_found)
    date_to_day_map = {}
    for idx, d in enumerate(dates_found):
        if idx == 0:
            date_to_day_map[d] = t["day_1"]
        elif idx == 1:
            date_to_day_map[d] = t["day_2"]
        else:
            date_to_day_map[d] = t["day_3"]

    for item in records:
        first_name = item.get("firstName", "")
        last_name = item.get("lastName", "")
        name = f"{first_name} {last_name}".strip()
        health = item.get("health", "ready")
        
        price_raw = item.get("fantasyPrice", 0) or 0
        price_m = round(price_raw / 1000000.0, 2)
        
        player_team_obj = item.get("team", {}) or {}
        positions_raw = player_team_obj.get("positions", ["guard"])
        pos_str = positions_raw[0] if len(positions_raw) > 0 else "guard"
        pos = pos_mapping.get(pos_str.lower(), "G")
        
        team_info = player_team_obj.get("team", {}) or {}
        translation = team_info.get("translation", {}) or {}
        team_name = translation.get("shortName") or translation.get("name") or "Unknown"
        
        game_day_str = t["day_1"]
        opp_team = "Unknown"
        
        games = team_info.get("games", [])
        for g in games:
            game_id = g.get("id")
            g_date = g.get("originalGameAt", "")[:10]
            if g_date in date_to_day_map:
                game_day_str = date_to_day_map[g_date]
                
            t1 = g.get("team1", {}).get("team", {}).get("translation", {}).get("shortName", "T1")
            t2 = g.get("team2", {}).get("team", {}).get("translation", {}).get("shortName", "T2")
            
            opp_team = t2 if team_name == t1 else t1
                
            if game_id and game_id not in schedule_dict:
                schedule_dict[game_id] = {
                    "Match": f"{t1} vs {t2}",
                    "Time": g.get("originalGameAt", "")[:16].replace("T", " "),
                    "Day": date_to_day_map.get(g_date, t["day_1"])
                }

        stats = item.get("stats") or {}
        gp = stats.get("s_gp", 0) or 0
        eff = stats.get("s_eff", 0) or 0
        avg_pir = item.get("average_fantasy_pts") or eff
        avg_pir = round(float(avg_pir), 1) if avg_pir is not None else 0.0
        
        # PROGNOZĖ TIK PAGAL VARŽOVO STIPRUMĄ
        proj_pir = calculate_advanced_proj(avg_pir, opp_team)
        value_score = round(proj_pir / price_m, 2) if price_m > 0 else 0.0

        players.append({
            t["col_name"]: name,
            t["col_team"]: team_name,
            t["col_opp"]: opp_team,
            t["col_pos"]: pos,
            t["col_price"]: price_m,
            t["col_pir"]: avg_pir,
            t["col_proj_pir"]: proj_pir,
            "Minutes": round((stats.get("s_time", 0) or 0) / 60.0 / max(gp, 1), 1) if gp > 0 else 0.0,
            t["col_val"]: value_score,
            t["col_day"]: game_day_str,
            t["col_status"]: health.capitalize()
        })
        
    return pd.DataFrame(players), pd.DataFrame(list(schedule_dict.values())) if schedule_dict else pd.DataFrame()

# ---------------------------------------------------------
# INPUT SELECTION
# ---------------------------------------------------------
input_option = st.sidebar.radio(t["source_select"], [t["src_sample"], t["src_json"]])

if input_option == t["src_json"]:
    json_text = st.sidebar.text_area("JSON Input:", height=180)
    df_players, df_schedule = parse_basketnews_json(json.loads(json_text)) if json_text.strip() else (pd.DataFrame(), pd.DataFrame())
else:
    sample_data = {
        "data": {
            "playersSearchRecordsFromClient": {
                "records": [
                    {"firstName":"Sasha","lastName":"Vezenkov","health":"ready","fantasyPrice":1950000,"team":{"positions":["forward"],"team":{"translation":{"shortName":"Olympiakos"},"games":[{"id":"g1","originalGameAt":"2026-10-07T18:15:00","team1":{"team":{"translation":{"shortName":"Olympiakos"}}},"team2":{"team":{"translation":{"shortName":"ALBA"}}}}]}},"stats":{"s_gp":3,"s_eff":24.5,"s_time":3600},"average_fantasy_pts":24.5},
                    {"firstName":"Kendrick","lastName":"Nunn","health":"ready","fantasyPrice":1650000,"team":{"positions":["guard"],"team":{"translation":{"shortName":"PAO"},"games":[{"id":"g2","originalGameAt":"2026-10-08T18:15:00","team1":{"team":{"translation":{"shortName":"Fenerbahce"}}},"team2":{"team":{"translation":{"shortName":"PAO"}}}}]}},"stats":{"s_gp":3,"s_eff":20.2,"s_time":3400},"average_fantasy_pts":20.2},
                    {"firstName":"Mike","lastName":"James","health":"ready","fantasyPrice":1850000,"team":{"positions":["guard"],"team":{"translation":{"shortName":"Monaco"},"games":[{"id":"g3","originalGameAt":"2026-10-09T19:00:00","team1":{"team":{"translation":{"shortName":"Monaco"}}},"team2":{"team":{"translation":{"shortName":"Real Madrid"}}}}]}},"stats":{"s_gp":3,"s_eff":22.8,"s_time":3550},"average_fantasy_pts":22.8},
                    {"firstName":"Mathias","lastName":"Lessort","health":"ready","fantasyPrice":1800000,"team":{"positions":["center"],"team":{"translation":{"shortName":"PAO"},"games":[{"id":"g2","originalGameAt":"2026-10-08T18:15:00","team1":{"team":{"translation":{"shortName":"Fenerbahce"}}},"team2":{"team":{"translation":{"shortName":"PAO"}}}}]}},"stats":{"s_gp":3,"s_eff":21.0,"s_time":3500},"average_fantasy_pts":21.0},
                    {"firstName":"Patrick","lastName":"Mills","health":"ready","fantasyPrice":1170000,"team":{"positions":["guard"],"team":{"translation":{"shortName":"ASVEL"},"games":[{"id":"g1","originalGameAt":"2026-10-07T18:45:00","team1":{"team":{"translation":{"shortName":"ASVEL"}}},"team2":{"team":{"translation":{"shortName":"Paris"}}}}]}},"stats":{"s_gp":3,"s_eff":16.8,"s_time":3650},"average_fantasy_pts":16.8},
                    {"firstName":"Bruno","lastName":"Caboclo","health":"ready","fantasyPrice":860000,"team":{"positions":["center"],"team":{"translation":{"shortName":"Hapoel"},"games":[{"id":"g3","originalGameAt":"2026-10-09T18:30:00","team1":{"team":{"translation":{"shortName":"Valencia"}}},"team2":{"team":{"translation":{"shortName":"Hapoel"}}}}]}},"stats":{"s_gp":1,"s_eff":12.1,"s_time":785},"average_fantasy_pts":12.1}
                ]
            }
        }
    }
    df_players, df_schedule = parse_basketnews_json(sample_data)

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
st.title(t["title"])
st.caption(t["subtitle"])

if not df_players.empty:
    tab_overview, tab_injuries, tab_schedule, tab_optimizer, tab_roster, tab_database = st.tabs([
        t["tab_overview"], t["tab_injuries"], t["tab_schedule"], t["tab_optimizer"], t["tab_roster"], t["tab_database"]
    ])

    with tab_optimizer:
        st.subheader(t["tab_optimizer"])
        max_budget = st.slider(t["budget"], min_value=3.0, max_value=15.0, value=9.0, step=0.1)
        
        if st.button(t["generate_opt"]):
            active = df_players[df_players[t["col_status"]].str.lower() == 'ready'].reset_index(drop=True)
            if len(active) > 0:
                c = -active[t["col_proj_pir"]].values
                A_ub, b_ub = [active[t["col_price"]].values], [max_budget]
                
                req_g = min(4, len(active[active[t["col_pos"]] == 'G']))
                req_f = min(4, len(active[active[t["col_pos"]] == 'F']))
                req_c = min(2, len(active[active[t["col_pos"]] == 'C']))
                
                A_eq = [
                    (active[t["col_pos"]] == 'G').astype(int).values,
                    (active[t["col_pos"]] == 'F').astype(int).values,
                    (active[t["col_pos"]] == 'C').astype(int).values,
                    np.ones(len(active))
                ]
                b_eq = [req_g, req_f, req_c, req_g + req_f + req_c]
                
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs')
                
                if res.success:
                    opt = active.iloc[np.where(res.x > 0.5)[0]].copy()
                    
                    d1_players = opt[opt[t["col_day"]] == t["day_1"]].sort_values(by=t["col_proj_pir"], ascending=False)
                    d2_players = opt[opt[t["col_day"]] == t["day_2"]].sort_values(by=t["col_proj_pir"], ascending=False)
                    d3_players = opt[opt[t["col_day"]] == t["day_3"]].sort_values(by=t["col_proj_pir"], ascending=False)
                    
                    c_d1 = d1_players.iloc[0] if not d1_players.empty else None
                    c_d2 = d2_players.iloc[0] if not d2_players.empty else None
                    c_d3 = d3_players.iloc[0] if not d3_players.empty else None
                    
                    st.success(t["opt_success"])
                    
                    col_cap1, col_cap2, col_cap3 = st.columns(3)
                    with col_cap1:
                        if c_d1 is not None:
                            st.info(f"### {t['c_day1']}\n**{c_d1[t['col_name']]}** ({c_d1[t['col_team']]})\n* Prognozuojamas PIR: **{c_d1[t['col_proj_pir']]}** (Vid: {c_d1[t['col_pir']]})\n* Varžovas: **vs {c_d1[t['col_opp']]}**")
                        else:
                            st.write("1-ąją dieną žaidėjų nėra.")
                            
                    with col_cap2:
                        if c_d2 is not None:
                            st.warning(f"### {t['c_day2']}\n**{c_d2[t['col_name']]}** ({c_d2[t['col_team']]})\n* Prognozuojamas PIR: **{c_d2[t['col_proj_pir']]}** (Vid: {c_d2[t['col_pir']]})\n* Varžovas: **vs {c_d2[t['col_opp']]}**")
                        else:
                            st.write("2-ąją dieną žaidėjų nėra.")

                    with col_cap3:
                        if c_d3 is not None:
                            st.error(f"### {t['c_day3']}\n**{c_d3[t['col_name']]}** ({c_d3[t['col_team']]})\n* Prognozuojamas PIR: **{c_d3[t['col_proj_pir']]}** (Vid: {c_d3[t['col_pir']]})\n* Varžovas: **vs {c_d3[t['col_opp']]}**")
                        else:
                            st.write("3-iąją dieną žaidėjų nėra.")
                    
                    st.markdown("---")
                    st.markdown("### 🧠 **Algoritminė Kapitonų Strategija**")
                    if c_d1 is not None:
                        thresh1 = max(18.0, round((c_d2[t['col_proj_pir']] if c_d2 is not None else 18.0) * 0.85, 1))
                        st.markdown(f"1️⃣ **PO 1-OSIOS DIENOS:** Jei **{c_d1[t['col_name']]}** surenka **$\ge$ {thresh1} PIR** $\\rightarrow$ **NEKEISTI KAPITONO**. Jei **$<$ {thresh1} PIR** $\\rightarrow$ perleisk kapitoną 2-osios dienos lyderiui.")
                    
                    if c_d2 is not None:
                        thresh2 = max(18.0, round((c_d3[t['col_proj_pir']] if c_d3 is not None else 18.0) * 0.85, 1))
                        st.markdown(f"2️⃣ **PO 2-OSIOS DIENOS (jei keitei po 1 d.):** Jei **{c_d2[t['col_name']]}** surenka **$\ge$ {thresh2} PIR** $\\rightarrow$ **NEKEISTI**. Jei **$<$ {thresh2} PIR** $\\rightarrow$ perleisk kapitoną 3-osios dienos lyderiui **{c_d3[t['col_name']] if c_d3 is not None else 'Nėra'}**!")
                    st.markdown("---")

                    st.dataframe(
                        opt[[t["col_name"], t["col_team"], t["col_opp"], t["col_pos"], t["col_price"], t["col_pir"], t["col_proj_pir"], t["col_day"], t["col_val"]]],
                        hide_index=True, use_container_width=True
                    )
                else:
                    st.error(t["opt_fail"])

    with tab_roster:
        st.subheader("📝 Mano sudėties vertinimas")
        st.caption("Pasirink savo žaidėjus — vertinimas skaičiuojamas pagal šio turo varžovą, prognozuojamą PIR ir žaidėjo būseną.")

        roster_options = df_players[t["col_name"]].tolist()
        selected_roster = st.multiselect(
            "👥 Pasirink savo sudėties žaidėjus:",
            options=roster_options,
            key="my_roster_selection"
        )

        if selected_roster:
            roster = df_players[df_players[t["col_name"]].isin(selected_roster)].copy()

            # ---------------------------------------------------------
            # EXACT BUDGET REPLACEMENT FINDER
            # ---------------------------------------------------------
            st.markdown("### 💰 Geriausi variantai pagal tavo tikslų biudžetą")
            st.caption(
                "Pasirink, kurį žaidėją nori keisti, ir įrašyk tiksliai, kiek €M gali skirti jo pakeitimui. "
                "Sistema ieškos tik įperkamų, READY ir tos pačios pozicijos žaidėjų šiame ture."
            )

            player_to_replace = st.selectbox(
                "🔄 Kurį žaidėją nori pakeisti?",
                options=selected_roster,
                key="budget_replace_player"
            )
            selected_player_row = roster[roster[t["col_name"]] == player_to_replace].iloc[0]
            current_price = float(selected_player_row[t["col_price"]])

            budget_col1, budget_col2 = st.columns(2)
            with budget_col1:
                exact_budget = st.number_input(
                    "💶 Kiek tiksliai gali išleisti? (€M)",
                    min_value=0.40,
                    max_value=10.00,
                    value=round(current_price, 2),
                    step=0.01,
                    format="%.2f",
                    key="exact_replacement_budget"
                )
            with budget_col2:
                st.metric(
                    "Dabartinė žaidėjo kaina",
                    f"€{current_price:.2f}M",
                    delta=f"€{exact_budget - current_price:+.2f}M"
                )

            if st.button("🔎 Rasti geriausius variantus", key="find_budget_options"):
                replace_pos = selected_player_row[t["col_pos"]]
                available_budget = df_players[
                    (df_players[t["col_name"]] != player_to_replace) &
                    (~df_players[t["col_name"]].isin(selected_roster)) &
                    (df_players[t["col_status"]].str.lower() == "ready") &
                    (df_players[t["col_pos"]] == replace_pos) &
                    (df_players[t["col_price"]] <= exact_budget)
                ].copy()

                if available_budget.empty:
                    st.warning(
                        f"Neradau tinkamo {replace_pos} pozicijos READY žaidėjo iki €{exact_budget:.2f}M."
                    )
                else:
                    available_budget["__upgrade"] = (
                        available_budget[t["col_proj_pir"]].astype(float) -
                        float(selected_player_row[t["col_proj_pir"]])
                    )
                    available_budget["__value"] = (
                        available_budget[t["col_proj_pir"]].astype(float) /
                        available_budget[t["col_price"]].astype(float).clip(lower=0.1)
                    )
                    available_budget["__budget_left"] = (
                        exact_budget - available_budget[t["col_price"]].astype(float)
                    )

                    # Pagrindinis reitingas: prognozuojamas PIR + pagerėjimas +
                    # vertė už €M. Brangesnis žaidėjas nėra automatiškai geresnis.
                    available_budget["__rank"] = (
                        available_budget[t["col_proj_pir"]].astype(float) * 1.0 +
                        available_budget["__upgrade"].clip(lower=0) * 1.25 +
                        available_budget["__value"] * 0.8
                    )
                    top_options = available_budget.sort_values("__rank", ascending=False).head(5)

                    st.success(
                        f"Radau {len(available_budget)} įperkamų variantų. "
                        f"Rodau geriausius {len(top_options)} pagal šio turo prognozę."
                    )

                    for rank, (_, option) in enumerate(top_options.iterrows(), start=1):
                        upgrade = round(float(option["__upgrade"]), 1)
                        budget_left = round(float(option["__budget_left"]), 2)
                        if upgrade > 0:
                            badge = "🟢 GERAS UPGRADE"
                        elif upgrade >= -1.0:
                            badge = "🟡 SAUGUS VARIANTAS"
                        else:
                            badge = "🔴 SILPNESNIS"

                        st.markdown(
                            f"**{rank}. {option[t["col_name"]]}** ({option[t["col_team"]]}) "
                            f"— €{option[t["col_price"]]:.2f}M | "
                            f"Prog. PIR **{option[t["col_proj_pir"]]:.1f}** | "
                            f"vs **{option[t["col_opp"]]}** | "
                            f"lieka **€{budget_left:.2f}M** | {badge} "
                            f"({upgrade:+.1f} PIR prieš {player_to_replace})"
                        )

                    st.info(
                        "💡 Pirmiausia žiūrėk į 🟢 variantus. Jei geriausias žaidėjas kainuoja mažiau "
                        "nei tavo biudžetas, nebūtina išleisti visų pinigų — svarbiau prognozuojamas PIR."
                    )
            roster["__score"] = roster[t["col_proj_pir"]].astype(float)

            avg_score = roster["__score"].mean()
            ready_ratio = (roster[t["col_status"]].str.lower() == "ready").mean()
            favorable_ratio = (roster["__score"] >= roster[t["col_pir"]]).mean()

            rating = 4.5 + min(3.0, avg_score / 8.0) + 1.5 * favorable_ratio + 1.0 * ready_ratio
            rating = round(max(1.0, min(10.0, rating)), 1)

            st.metric("🏆 Tavo sudėties įvertinimas", f"{rating}/10")

            def classify(row):
                proj = float(row[t["col_proj_pir"]])
                base = float(row[t["col_pir"]])
                status = str(row[t["col_status"]]).lower()

                if status != "ready" or proj < base * 0.82:
                    return "🔴 BŪTINA KEISTI"
                if proj < base * 0.97:
                    return "🟡 ANT KLAUSTUKO"
                return "🟢 VISIŠKAI GERAS"

            roster["__category"] = roster.apply(classify, axis=1)

            must_change = roster[roster["__category"] == "🔴 BŪTINA KEISTI"]
            question = roster[roster["__category"] == "🟡 ANT KLAUSTUKO"]
            good = roster[roster["__category"] == "🟢 VISIŠKAI GERAS"]

            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 Būtina keisti", len(must_change))
            c2.metric("🟡 Ant klaustuko", len(question))
            c3.metric("🟢 Visiškai geri", len(good))

            # ---------------------------------------------------------
            # SMART REPLACEMENT ENGINE
            # ---------------------------------------------------------
            # Ieško realios alternatyvos panašioje kainoje. Pirmenybė:
            # 1) žaidėjui, kurio prognozuojamas PIR yra aukštesnis;
            # 2) pasirinktame ture jis turi būti READY;
            # 3) ta pati pozicija;
            # 4) panaši kaina (±20%, su minimaliu tolerancijos lygiu).
            def find_replacement(player, roster_names):
                price = float(player[t["col_price"]])
                pos = player[t["col_pos"]]
                proj = float(player[t["col_proj_pir"]])

                available = df_players[
                    (df_players[t["col_name"]] != player[t["col_name"]]) &
                    (~df_players[t["col_name"]].isin(roster_names)) &
                    (df_players[t["col_status"]].str.lower() == "ready") &
                    (df_players[t["col_pos"]] == pos)
                ].copy()

                if available.empty:
                    return None

                # Panaši kaina: pirmiausia ±20%, bet pigesniems žaidėjams
                # leidžiame šiek tiek platesnį diapazoną.
                min_price = max(0.4, price * 0.80)
                max_price = price * 1.20
                similar = available[
                    (available[t["col_price"]] >= min_price) &
                    (available[t["col_price"]] <= max_price)
                ].copy()

                # Jei nėra tinkamo žaidėjo ±20%, ieškome ±30%.
                if similar.empty:
                    min_price = max(0.4, price * 0.70)
                    max_price = price * 1.30
                    similar = available[
                        (available[t["col_price"]] >= min_price) &
                        (available[t["col_price"]] <= max_price)
                    ].copy()

                if similar.empty:
                    return None

                # Nenorime siūlyti žaidėjo, kuris šiame ture yra akivaizdžiai
                # blogesnis už keičiamą žaidėją.
                better = similar[similar[t["col_proj_pir"]] > proj].copy()
                candidates = better if not better.empty else similar

                # Balas vertina prognozę, kainos panašumą ir pranašumą prieš
                # keičiamą žaidėją. Taip neparenkamas tiesiog pats brangiausias.
                candidates["__price_diff"] = (
                    abs(candidates[t["col_price"]] - price) / max(price, 0.1)
                )
                candidates["__upgrade"] = (
                    candidates[t["col_proj_pir"]] - proj
                )
                candidates["__replacement_score"] = (
                    candidates[t["col_proj_pir"]] * 1.0 +
                    candidates["__upgrade"] * 0.8 -
                    candidates["__price_diff"] * 5.0
                )

                return candidates.sort_values(
                    "__replacement_score", ascending=False
                ).iloc[0]

            def render_group(title, data, show_replacement=False):
                st.markdown(f"### {title}")
                if data.empty:
                    st.success("Šioje kategorijoje žaidėjų nėra.")
                    return

                roster_names = set(selected_roster)

                for _, r in data.sort_values("__score", ascending=False).iterrows():
                    delta = round(
                        float(r[t["col_proj_pir"]]) - float(r[t["col_pir"]]), 1
                    )
                    st.markdown(
                        f"**{r[t['col_name']]}** ({r[t['col_team']]}) — "
                        f"Prog. PIR **{r[t['col_proj_pir']]}** vs vid. **{r[t['col_pir']]}** | "
                        f"vs **{r[t['col_opp']]}** | {r[t['col_day']]} | "
                        f"pokytis **{delta:+.1f}**"
                    )

                    if show_replacement:
                        replacement = find_replacement(r, roster_names)

                        if replacement is not None:
                            price_diff = round(
                                float(replacement[t["col_price"]]) -
                                float(r[t["col_price"]]), 2
                            )
                            upgrade = round(
                                float(replacement[t["col_proj_pir"]]) -
                                float(r[t["col_proj_pir"]]), 1
                            )

                            st.success(
                                f"🔄 **Keisčiau į: {replacement[t['col_name']]}** "
                                f"({replacement[t['col_team']]}) — "
                                f"kaina €{replacement[t['col_price']]:.2f}M "
                                f"({'+' if price_diff >= 0 else ''}{price_diff:.2f}M), "
                                f"prog. PIR **{replacement[t['col_proj_pir']]}** "
                                f"({'+' if upgrade >= 0 else ''}{upgrade} PIR). "
                                f"Varžovas: **vs {replacement[t['col_opp']]}**."
                            )
                        else:
                            st.info(
                                "🔄 Neradau pakankamai gero kandidato panašioje "
                                "kainoje ir toje pačioje pozicijoje."
                            )

            render_group(
                "🔴 Žaidėjai, kuriuos būtinai keisčiau",
                must_change,
                show_replacement=True
            )
            render_group(
                "🟡 Žaidėjai ant klaustuko",
                question,
                show_replacement=True
            )
            render_group(
                "🟢 Žaidėjai, kuriuos palikčiau",
                good,
                show_replacement=False
            )

            st.markdown("### 📊 Visa tavo sudėtis")
            st.dataframe(
                roster[
                    [t["col_name"], t["col_team"], t["col_opp"], t["col_pos"],
                     t["col_pir"], t["col_proj_pir"], t["col_day"], t["col_status"], "__category"]
                ].rename(columns={"__category": "Vertinimas"}),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Pasirink savo žaidėjus aukščiau, kad gaučiau tavo sudėties įvertinimą iš 10.")

    with tab_injuries:
        st.subheader(t["injury_sim"])
        all_teams = df_players[t["col_team"]].unique().tolist()
        sel_team = st.selectbox("Komanda / Team:", all_teams)
        team_df = df_players[df_players[t["col_team"]] == sel_team].copy()
        
        out_sel = st.multiselect(t["out_players"], options=team_df[t["col_name"]].tolist())
        
        if out_sel:
            active_m = ~team_df[t["col_name"]].isin(out_sel)
            lost_pir = team_df[team_df[t["col_name"]].isin(out_sel)][t["col_pir"]].sum()
            if active_m.sum() > 0:
                team_df.loc[active_m, t["col_proj_pir"]] = team_df.loc[active_m, t["col_pir"]] + round((lost_pir * 0.45) / active_m.sum(), 1)
        
        st.dataframe(team_df[[t["col_name"], t["col_pos"], t["col_price"], t["col_pir"], t["col_proj_pir"], t["col_day"]]], hide_index=True, use_container_width=True)

    with tab_schedule:
        st.subheader(t["tab_schedule"])
        if not df_schedule.empty:
            c1, c2, c3 = st.columns(3)
            c1.write(f"### 📅 {t['day_1']}")
            c1.dataframe(df_schedule[df_schedule['Day'] == t['day_1']][['Match', 'Time']], hide_index=True, use_container_width=True)
            c2.write(f"### 📅 {t['day_2']}")
            c2.dataframe(df_schedule[df_schedule['Day'] == t['day_2']][['Match', 'Time']], hide_index=True, use_container_width=True)
            c3.write(f"### 📅 {t['day_3']}")
            c3.dataframe(df_schedule[df_schedule['Day'] == t['day_3']][['Match', 'Time']], hide_index=True, use_container_width=True)

    with tab_overview:
        m1, m2, m3 = st.columns(3)
        m1.metric(t["total_players"], len(df_players))
        m2.metric(t["avg_pir"], f"{df_players[t['col_pir']].mean():.1f}")
        m3.metric(t["top_val"], df_players.sort_values(by=t["col_val"], ascending=False).iloc[0][t["col_name"]])
        st.dataframe(df_players, hide_index=True, use_container_width=True)

    with tab_database:
        st.dataframe(df_players, hide_index=True, use_container_width=True)