import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Model wzrostu i zysku", page_icon="📈", layout="wide")
px.defaults.template = "plotly_white"


st.markdown("""
<style>
	.stApp,
	[data-testid="stMarkdownContainer"],
	[data-testid="stMetricLabel"],
	[data-testid="stMetricValue"],
	[data-testid="stCaptionContainer"],
	.stText,
	h1, h2, h3, h4, h5, h6,
	label {
		color: #1f2937 !important;
	}

	[data-testid="stSidebar"] {
		color: #1f2937;
	}
</style>
""", unsafe_allow_html=True)


def solve_alpha(capital, labor, params, alk):
	"""Solve alpha so marginal KLR equals the stock ratio K/L."""
	def residual(candidate):
		beta = 1 - candidate
		output = (
			params["q0"] * params["E"] ** params["zeta"]
			* (capital / params["K0"]) ** candidate
			* (labor / params["L0"]) ** beta
		)
		return candidate - capital / max(output, 1e-12) * (1 / alk + params["R"])

	lower, upper = 0.02, 0.98
	if residual(lower) * residual(upper) > 0:
		raise ValueError("Nie można wyznaczyć alpha w zakresie 0.02-0.98 dla podanych parametrów.")
	for _ in range(80):
		middle = (lower + upper) / 2
		if residual(lower) * residual(middle) <= 0:
			upper = middle
		else:
			lower = middle
	return (lower + upper) / 2


def simulate(params):
	"""Run the model with market-clearing investment."""
	dt = params["dt"]
	steps = int(round(params["periods"] / dt))
	times = np.arange(steps + 1) * dt
	capital = np.zeros(len(times))
	labor = np.zeros(len(times))
	population = np.zeros(len(times))
	output = np.zeros(len(times))
	alpha = np.zeros(len(times))
	beta = np.zeros(len(times))
	alk = np.zeros(len(times))
	kor = np.zeros(len(times))
	klr = np.zeros(len(times))
	wage = np.zeros(len(times))
	consumption = np.zeros(len(times))
	investment = np.zeros(len(times))
	government = np.zeros(len(times))
	exports = np.zeros(len(times))
	imports = np.zeros(len(times))
	profit = np.zeros(len(times))
	klr_formula = np.zeros(len(times))
	klr_difference = np.zeros(len(times))
	planned_demand = np.zeros(len(times))
	planned_investment_series = np.zeros(len(times))
	demand = np.zeros(len(times))

	capital[0] = params["K0"]
	labor[0] = params["L0"]
	population[0] = params["Pop0"]
	employment_rate = params["L0"] / params["Pop0"]

	for t in range(steps + 1):
		population[t] = params["Pop0"] * np.exp(params["n"] * times[t])
		labor[t] = employment_rate * population[t]
		if t == 0:
			alk[t] = params["alk0"]
		else:
			adjustment = (
				- params["gamma0"]
				- params["gamma_E"] * (1 - params["E"])
			) * (klr[t - 1] / params["KLR0"] - 1)
			alk[t] = params["alk0"] * np.exp(adjustment)
		alpha[t] = solve_alpha(capital[t], labor[t], params, alk[t])
		beta[t] = 1 - alpha[t]

		output[t] = (
			params["q0"] * params["E"] ** params["zeta"]
			* (capital[t] / params["K0"]) ** alpha[t]
			* (labor[t] / params["L0"]) ** beta[t]
		)
		kor[t] = capital[t] / max(output[t], 1e-12)
		wage[t] = beta[t] * output[t] / max(labor[t], 1e-12)
		klr[t] = capital[t] / max(labor[t], 1e-12)
		klr_formula[t] = (
			alpha[t] * wage[t]
			/ max((1 / alk[t] + params["R"]) * beta[t], 1e-12)
		)
		klr_difference[t] = klr[t] - klr_formula[t]

		consumption[t] = params["a"] * params["Pq"] * output[t]
		government[t] = params["GovSp"] * params["Pq"] * output[t]
		imports[t] = params["m0"] * params["Pq"] * output[t] * (params["alk0"] / max(alk[t], 1e-12)) ** params["eta"]
		exports[t] = params["x0"] * params["Pq"] * output[t] * (klr[t] / params["KLR0"]) ** params["kappa"]
		planned_investment = max(params["b"] * (consumption[t] - consumption[t - 1]), 0) if t > 0 else 0
		planned_investment_series[t] = planned_investment
		planned_demand[t] = consumption[t] + planned_investment + government[t] + exports[t] - imports[t]
		investment[t] = planned_investment + params["Pq"] * output[t] - planned_demand[t]
		demand[t] = consumption[t] + investment[t] + government[t] + exports[t] - imports[t]


		profit[t] = output[t] * params["Pq"] - capital[t] * params["Pk"] * (1 / alk[t] + params["R"]) - labor[t] * wage[t] * params["Pq"]

		if t < steps:
			capital[t + 1] = max(capital[t] + dt * (investment[t] / params["Pq"] - capital[t] / max(alk[t], 1e-12)), 1e-8)

	klr_growth = np.r_[np.nan, np.diff(klr) / np.maximum(klr[:-1], 1e-12) / dt]
	wage_growth = np.r_[np.nan, np.diff(wage) / np.maximum(wage[:-1], 1e-12) / dt]
	capital_growth = np.r_[np.nan, np.diff(capital) / np.maximum(capital[:-1], 1e-12) / dt]
	labor_growth = np.r_[np.nan, np.diff(labor) / np.maximum(labor[:-1], 1e-12) / dt]
	production_growth = np.r_[np.nan, np.diff(output) / np.maximum(output[:-1], 1e-12) / dt]
	growth_identity = (
		alpha * capital_growth
		+ beta * labor_growth
		+ kor * wage / np.maximum(klr, 1e-12)
		* np.log(np.maximum(klr / params["KLR0"], 1e-12))
		* (klr_growth - wage_growth)
	)

	return pd.DataFrame({
		"Okres": times, "Kapitał K": capital, "Praca L": labor, "Populacja": population,
		"Produkcja q": output, "Popyt planowany Yp": planned_demand, "Popyt Y": demand, "alpha": alpha,
		"beta": beta, "alk": alk, "KOR": kor, "KLR": klr, "Płaca rw": wage,
		"Konsumpcja C": consumption, "Inwestycje I": investment,
		"Inwestycje planowane Ip": planned_investment_series,
		"Wydatki G": government,
		"Eksport X": exports, "Import M": imports, "Zysk pi": profit,
		"Luka popytowa": demand - params["Pq"] * output, "Wzrost gospodarczy": growth_identity,
	})


def base_params(values):
	names = ["periods", "dt", "K0", "L0", "Pop0", "q0", "alk0", "E", "zeta", "gamma_E", "gamma0", "R", "n", "a", "b", "GovSp", "m0", "eta", "x0", "kappa", "Pq", "Pk"]
	params = {name: values[name] for name in names}
	params["alpha0"] = solve_alpha(params["K0"], params["L0"], params, params["alk0"])
	params["beta0"] = 1 - params["alpha0"]
	params["wage0"] = params["beta0"] * params["q0"] / params["L0"]
	params["KLR0"] = params["K0"] / params["L0"]
	return params


def scenario_inputs(label, key_prefix):
	st.subheader(label)
	periods = st.number_input("Horyzont symulacji", 5, 200, 40, key=f"{key_prefix}_periods")
	K0 = st.number_input("Kapitał K₀", 1.0, 1_000_000.0, 100.0, key=f"{key_prefix}_K0")
	L0 = st.number_input("Praca L₀", 1.0, 1_000_000.0, 100.0, key=f"{key_prefix}_L0")
	Pop0 = st.number_input("Populacja Pop₀", 1.0, 10_000_000.0, 108.0, key=f"{key_prefix}_Pop0_v2")
	q0 = st.number_input("Produkcja bazowa q₀", 1.0, 1_000_000.0, 100.0, key=f"{key_prefix}_q0")
	alk0 = st.number_input("Średni okres użytkowania alk₀", 1.0, 100.0, 10.0, key=f"{key_prefix}_alk0_v2")
	st.subheader("Technologia i dynamika")
	if f"{key_prefix}_E" in st.session_state and not 0 < st.session_state[f"{key_prefix}_E"] <= 1:
		st.session_state[f"{key_prefix}_E"] = 1.0
	if f"{key_prefix}_zeta" in st.session_state and st.session_state[f"{key_prefix}_zeta"] <= 0:
		st.session_state[f"{key_prefix}_zeta"] = 0.10
	E = st.slider("Współczynnik oddziaływania transformacji ekologicznej E", 0.01, 1.0, 1.0, 0.01, key=f"{key_prefix}_E")
	zeta = st.number_input("ζ", 0.01, 5.0, 1.0, 0.1, key=f"{key_prefix}_zeta")
	gamma_E = st.number_input("γ_E", -5.0, 5.0, 0.25, 0.05, key=f"{key_prefix}_gamma_E")
	gamma0 = st.number_input("γ₀", -5.0, 5.0, 0.25, 0.05, key=f"{key_prefix}_gamma0")
	R = st.slider("R", -0.05, 0.50, 0.05, 0.01, key=f"{key_prefix}_R")
	alpha0 = (K0 / (q0 * E ** zeta)) * (1 / alk0 + R)
	st.metric("Bazowe KLR₀", f"{K0 / L0:.4f}")
	st.caption(f"α₀ jest wyliczane ze wzoru i nie można go ustawić ręcznie: α₀ = {alpha0:.4f}")
	n = st.slider("Tempo wzrostu populacji n", -0.05, 0.10, 0.02, 0.005, key=f"{key_prefix}_n")
	st.subheader("Popyt i handel")
	a = st.slider("Skłonność do konsumpcji a", 0.0, 1.0, 0.70, 0.01, key=f"{key_prefix}_a")
	b = st.slider("Parametr inwestycji b", 0.0, 5.0, 1.00, 0.05, key=f"{key_prefix}_b")
	GovSp = st.slider("Wydatki rządowe / Y", 0.0, 0.5, 0.15, 0.01, key=f"{key_prefix}_GovSp")
	m0 = st.slider("Import / Y (m₀)", 0.0, 0.8, 0.20, 0.01, key=f"{key_prefix}_m0")
	eta = st.number_input("η", -5.0, 5.0, 0.5, 0.1, key=f"{key_prefix}_eta")
	x0 = st.slider("Eksport / Y (x₀)", 0.0, 0.8, 0.20, 0.01, key=f"{key_prefix}_x0")
	kappa = st.number_input("κ", -5.0, 5.0, 0.5, 0.1, key=f"{key_prefix}_kappa")
	Pq = st.number_input("Cena produktu Pq", 0.01, 100.0, 1.0, 0.1, key=f"{key_prefix}_Pq")
	Pk = st.number_input("Cena kapitału Pk", 0.01, 100.0, 1.0, 0.1, key=f"{key_prefix}_Pk")
	return {
		"periods": periods, "dt": 0.01, "K0": K0, "L0": L0, "Pop0": Pop0,
		"q0": q0, "alk0": alk0, "E": E, "zeta": zeta,
		"R": R, "n": n, "a": a, "b": b, "gamma_E": gamma_E, "gamma0": gamma0,
		"GovSp": GovSp, "m0": m0, "eta": eta, "x0": x0, "kappa": kappa,
		"Pq": Pq, "Pk": Pk,
	}


st.title("Model wzrostu gospodarczego z uwzględnieniem transformacji ekologicznej")
st.caption("Symulacja dynamiki produkcji i czynników wzrostu z uwzględnieniem transformacji ekologicznej.")

with st.sidebar:
	st.header("Parametry symulacji")
	st.caption("Krok obliczeń: dt = 0.01")
	st.session_state["scenario_a_E"] = 1.0
	st.session_state["scenario_a_zeta"] = 1.0
	with st.expander("Scenariusz A", expanded=True):
		values = scenario_inputs("Parametry scenariusza A", "scenario_a")
	compare_scenarios = st.checkbox("Pokaż scenariusz B na wspólnych wykresach", value=True)
	if compare_scenarios:
		with st.expander("Scenariusz B", expanded=True):
			st.caption("Scenariusz B dziedziczy parametry A z możliwością zmiany E, ζ, γ₀ i γ_E.")
			values_b = dict(values)
			if "scenario_b_E" in st.session_state and not 0 < st.session_state["scenario_b_E"] < 1:
				st.session_state["scenario_b_E"] = 0.70
			values_b["E"] = st.slider("Czynnik środowiskowy E", 0.01, 0.99, 0.70, 0.01, key="scenario_b_E")
			values_b["zeta"] = st.number_input("ζ", 0.01, 5.0, 1.2, 0.1, key="scenario_b_zeta")
			values_b["gamma0"] = st.number_input("γ₀", 0.0, 5.0, 0.25, 0.05, key="scenario_b_gamma0")
			values_b["gamma_E"] = st.number_input("γ_E", 0.0, 5.0, 0.25, 0.05, key="scenario_b_gamma_E")
		with st.expander("Scenariusz C", expanded=True):
			st.caption("C pokazuje korzystny wariant: niższe E i niższe ζ zwiększają poziom produkcji.")
			values_c = dict(values)
			if "scenario_c_E" in st.session_state and not 0 < st.session_state["scenario_c_E"] < 1:
				st.session_state["scenario_c_E"] = 0.25
			values_c["E"] = st.slider("Czynnik środowiskowy E", 0.01, 0.99, 0.85, 0.01, key="scenario_c_E")
			if "scenario_c_zeta" in st.session_state and st.session_state["scenario_c_zeta"] <= 0:
				st.session_state["scenario_c_zeta"] = 1.2
			values_c["zeta"] = st.number_input("ζ", 0.01, 5.0, 1.2, 0.1, key="scenario_c_zeta")
			values_c["gamma0"] = st.number_input("γ₀", 0.0, 5.0, 0.25, 0.05, key="scenario_c_gamma0")
			values_c["gamma_E"] = st.number_input("γ_E", 0.0, 5.0, 0.25, 0.05, key="scenario_c_gamma_E")
	else:
		values_b = None
		values_c = None

scenario_data = {"Scenariusz A": simulate(base_params(values))}
if values_b is not None:
	scenario_data["Scenariusz B"] = simulate(base_params(values_b))
if values_c is not None:
	scenario_data["Scenariusz C"] = simulate(base_params(values_c))
data = scenario_data["Scenariusz A"]

def get_axis_label(column):
    units = {
        "Kapitał K": "Kapitał K [j.u.]",
        "Praca L": "Praca L [j.u.]",
        "Populacja": "Populacja Pop [j.u.]",
        "Produkcja q": "Produkcja q [j.u.]",
        "Popyt planowany Yp": "Popyt planowany Yᵖ [j.u.]",
        "Popyt Y": "Popyt Y [j.u.]",
        "alpha": "Udział kapitału α [-]",
        "beta": "Udział pracy β [-]",
        "alk": "Średni okres użytkowania kapitału alk [lata]",
        "KOR": "Relacja kapitału do produkcji KOR [j.u.]",
        "KLR": "Techniczne uzbrojenie pracy KLR [j.u.]",
        "Płaca rw": "Płaca rw [j.u.]",
        "Konsumpcja C": "Konsumpcja C [j.u.]",
        "Inwestycje I": "Inwestycje faktyczne I [j.u.]",
        "Inwestycje planowane Ip": "Inwestycje planowane Iᵖ [j.u.]",
        "Wydatki G": "Wydatki rządowe G [j.u.]",
        "Eksport X": "Eksport X [j.u.]",
        "Import M": "Import M [j.u.]",
        "Zysk pi": "Zysk π [j.u.]",
        "Luka popytowa": "Luka popytowa [j.u.]",
        "Wzrost gospodarczy": "Wzrost gospodarczy [-]",
    }

    return units.get(column, column)


def get_chart_title(column):
    titles = {
        "Kapitał K": "Dynamika zasobu kapitału K",
        "Praca L": "Dynamika zasobu pracy L",
        "Populacja": "Dynamika populacji Pop",
        "Produkcja q": "Dynamika produkcji q",
        "Popyt planowany Yp": "Dynamika popytu planowanego Yᵖ",
        "Popyt Y": "Dynamika popytu Y",
        "alpha": "Dynamika udziału kapitału w produkcji α",
        "beta": "Dynamika udziału pracy w produkcji β",
        "alk": "Dynamika średniego okresu użytkowania kapitału alk",
        "KOR": "Dynamika relacji kapitału do produkcji KOR",
        "KLR": "Dynamika technicznego uzbrojenia pracy KLR",
        "Płaca rw": "Dynamika płacy rw",
        "Konsumpcja C": "Dynamika konsumpcji C",
        "Inwestycje I": "Dynamika inwestycji faktycznych I",
        "Inwestycje planowane Ip": "Dynamika inwestycji planowanych Iᵖ",
        "Wydatki G": "Dynamika wydatków rządowych G",
        "Eksport X": "Dynamika eksportu X",
        "Import M": "Dynamika importu M",
        "Zysk pi": "Dynamika zysku π",
        "Luka popytowa": "Dynamika luki popytowej",
        "Wzrost gospodarczy": "Wzrost gospodarczy",
    }

    return titles.get(column, column)


def scenario_chart(column, title=None):
    frames = []

    for scenario_name, frame in scenario_data.items():
        part = frame[["Okres", column]].copy()
        part["Scenariusz"] = scenario_name
        frames.append(part)

    chart_data = pd.concat(frames, ignore_index=True)

    fig = px.line(
        chart_data,
        x="Okres",
        y=column,
        color="Scenariusz",
        title=title or get_chart_title(column)
    )

    fig.update_layout(
    title=dict(
        text=get_chart_title(column),
        y=0.85,
        yanchor="top",
        font=dict(color="#1f2937")
    ),
    font=dict(color="#1f2937"),
    legend_font=dict(color="#1f2937"),
    legend_title_font=dict(color="#1f2937")
)

    fig.update_xaxes(
        title_text="Czas [lata]",
        title_font=dict(color="#1f2937"),
        tickfont=dict(color="#1f2937")
    )

    fig.update_yaxes(
        title_text=get_axis_label(column),
        title_font=dict(color="#1f2937"),
        tickfont=dict(color="#1f2937")
    )

    return fig


def sensitivity_chart(variant_data, column, title):
    frames = []

    for scenario_name, frame in variant_data.items():
        part = frame[["Okres", column]].copy()
        part["Scenariusz"] = scenario_name
        frames.append(part)

    chart_data = pd.concat(frames, ignore_index=True)

    fig = px.line(
        chart_data,
        x="Okres",
        y=column,
        color="Scenariusz",
        title=title
    )

    fig.update_layout(
    title=dict(
        text=title,
        y=0.85,
        yanchor="top",
        font=dict(color="#1f2937")
    ),
    font=dict(color="#1f2937"),
    legend_font=dict(color="#1f2937"),
    legend_title_font=dict(color="#1f2937")
)
    fig.update_xaxes(
        title_text="Czas [lata]",
        title_font=dict(color="#1f2937"),
        tickfont=dict(color="#1f2937")
    )

    fig.update_yaxes(
        title_text=get_axis_label(column),
        title_font=dict(color="#1f2937"),
        tickfont=dict(color="#1f2937")
    )

    return fig


sensitivity_bases = {
    "Scenariusz A": values,
    "Scenariusz B": values_b if values_b is not None else {
        **values,
        "E": 0.70,
        "zeta": 1.2,
        "gamma0": 0.25,
        "gamma_E": 0.25,
    },
    "Scenariusz C": values_c if values_c is not None else {
        **values,
        "E": 0.85,
        "zeta": 1.2,
        "gamma0": 0.25,
        "gamma_E": 0.25,
    },
}


sensitivity_data = {}

for parameter, variants in {
    "K0": [
        ("K0 = 50", 50.0),
        ("K0 = 200", 200.0)
    ],
    "alk0": [
        ("alk0 = 5", 5.0),
        ("alk0 = 20", 20.0)
    ],
    "a": [
        ("a = 0,60", 0.60),
        ("a = 0,80", 0.80)
    ],
}.items():

    for variant_name, variant_value in variants:
        sensitivity_data[variant_name] = {}

        for scenario_name, scenario_values in sensitivity_bases.items():
            variant_values = dict(scenario_values)
            variant_values[parameter] = variant_value

            sensitivity_data[variant_name][scenario_name] = simulate(
                base_params(variant_values)
            )


metric_definitions = [
    (
        "Produkcja końcowa",
        lambda frame: f"{frame['Produkcja q'].iloc[-1]:,.2f}"
    ),
    (
        "Kapitał końcowy",
        lambda frame: f"{frame['Kapitał K'].iloc[-1]:,.2f}"
    ),
    (
        "α końcowe",
        lambda frame: f"{frame['alpha'].iloc[-1]:.3f}"
    ),
    (
        "Suma zysku",
        lambda frame: f"{frame['Zysk pi'].sum():,.2f}"
    ),
]


metric_cols = st.columns(len(metric_definitions))

for metric_column, (label, formatter) in zip(
    metric_cols,
    metric_definitions
):
    with metric_column:
        st.markdown(f"**{label}**")

        for scenario_name, frame in scenario_data.items():
            st.write(
                f"{scenario_name}: {formatter(frame)}"
            )


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Przebieg modelu",
    "Równowaga popytu",
    "Analiza wrażliwości",
    "Założenia",
    "Wzory",
])


with tab1:
    st.subheader("Osobne wykresy wszystkich zmiennych")

    plot_columns = [
        column
        for column in data.columns
        if column not in {
            "Okres",
            "Różnica KLR",
            "Bezrobocie u"
        }
    ]

    for index in range(0, len(plot_columns), 2):
        left, right = st.columns(2)

        with left:
            column = plot_columns[index]

            st.plotly_chart(
                scenario_chart(column),
                use_container_width=True,
                key=f"all_variables_left_{index}_{column}"
            )

        if index + 1 < len(plot_columns):
            with right:
                column = plot_columns[index + 1]

                st.plotly_chart(
                    scenario_chart(column),
                    use_container_width=True,
                    key=f"all_variables_right_{index}_{column}"
                )

    st.dataframe(
        data.round(4),
        use_container_width=True,
        hide_index=True
    )


with tab2:
    accounting_gap = (
        data["Produkcja q"] * values["Pq"] - data["Popyt Y"]
    ).abs().max()

    if accounting_gap < 1e-10:
        st.success(
            "Równowaga rynkowa spełniona: produkcja = popyt w każdym okresie."
        )
    else:
        st.warning(
            f"Równowaga rynkowa niespełniona. Luka: {accounting_gap:.2e}"
        )

    st.write(
        "Popyt planowany wykorzystuje behawioralną inwestycję "
        "Iᵖ = b · ΔC. Inwestycja faktyczna jest korygowana o różnicę "
        "między produkcją a popytem planowanym, aby zapewnić równowagę rynkową."
    )

    st.plotly_chart(
        scenario_chart(
            "Inwestycje I",
            "Dynamika inwestycji faktycznych"
        ),
        use_container_width=True,
        key="investment_chart"
    )

    st.plotly_chart(
        scenario_chart(
            "Wzrost gospodarczy",
            "Wzrost gospodarczy"
        ),
        use_container_width=True,
        key="output_growth_chart"
    )


with tab3:
    st.subheader(
        "Wpływ wybranych parametrów na dynamikę modelu"
    )

    st.caption(
        "Każdy wariant zmienia wyłącznie parametr wskazany w tytule; "
        "pozostałe parametry pozostają takie jak w scenariuszu A."
    )

    sensitivity_charts = [
        (
            "K0 = 50",
            "Kapitał K",
            "Dynamika zasobu kapitału dla K₀ = 50"
        ),
        (
            "K0 = 200",
            "Kapitał K",
            "Dynamika zasobu kapitału dla K₀ = 200"
        ),

        (
            "K0 = 50",
            "Inwestycje I",
            "Dynamika inwestycji faktycznych dla K₀ = 50"
        ),
        (
            "K0 = 200",
            "Inwestycje I",
            "Dynamika inwestycji faktycznych dla K₀ = 200"
        ),

        (
            "K0 = 50",
            "KLR",
            "Dynamika technicznego uzbrojenia pracy dla K₀ = 50"
        ),
        (
            "K0 = 200",
            "KLR",
            "Dynamika technicznego uzbrojenia pracy dla K₀ = 200"
        ),

        (
            "K0 = 50",
            "alk",
            "Dynamika średniego okresu użytkowania kapitału dla K₀ = 50"
        ),
        (
            "K0 = 200",
            "alk",
            "Dynamika średniego okresu użytkowania kapitału dla K₀ = 200"
        ),

        (
            "K0 = 50",
            "Produkcja q",
            "Dynamika produkcji dla K₀ = 50"
        ),
        (
            "K0 = 200",
            "Produkcja q",
            "Dynamika produkcji dla K₀ = 200"
        ),

        (
            "K0 = 50",
            "Wzrost gospodarczy",
            "Wzrost gospodarczy dla K₀ = 50"
        ),
        (
            "K0 = 200",
            "Wzrost gospodarczy",
            "Wzrost gospodarczy dla K₀ = 200"
        ),

        (
            "alk0 = 5",
            "Kapitał K",
            "Dynamika zasobu kapitału dla alk₀ = 5"
        ),
        (
            "alk0 = 20",
            "Kapitał K",
            "Dynamika zasobu kapitału dla alk₀ = 20"
        ),

        (
            "alk0 = 5",
            "KLR",
            "Dynamika technicznego uzbrojenia pracy dla alk₀ = 5"
        ),
        (
            "alk0 = 20",
            "KLR",
            "Dynamika technicznego uzbrojenia pracy dla alk₀ = 20"
        ),

        (
            "alk0 = 5",
            "alk",
            "Dynamika średniego okresu użytkowania kapitału dla alk₀ = 5"
        ),
        (
            "alk0 = 20",
            "alk",
            "Dynamika średniego okresu użytkowania kapitału dla alk₀ = 20"
        ),

        (
            "alk0 = 5",
            "Wzrost gospodarczy",
            "Wzrost gospodarczy dla alk₀ = 5"
        ),
        (
            "alk0 = 20",
            "Wzrost gospodarczy",
            "Wzrost gospodarczy dla alk₀ = 20"
        ),

        (
            "a = 0,60",
            "Inwestycje I",
            "Dynamika inwestycji faktycznych dla a = 0,60"
        ),
        (
            "a = 0,80",
            "Inwestycje I",
            "Dynamika inwestycji faktycznych dla a = 0,80"
        ),

        (
            "a = 0,60",
            "Inwestycje planowane Ip",
            "Dynamika inwestycji planowanych dla a = 0,60"
        ),
        (
            "a = 0,80",
            "Inwestycje planowane Ip",
            "Dynamika inwestycji planowanych dla a = 0,80"
        ),

        (
            "a = 0,60",
            "Kapitał K",
            "Dynamika zasobu kapitału dla a = 0,60"
        ),
        (
            "a = 0,80",
            "Kapitał K",
            "Dynamika zasobu kapitału dla a = 0,80"
        ),

        (
            "a = 0,60",
            "Wzrost gospodarczy",
            "Wzrost gospodarczy dla a = 0,60"
        ),
        (
            "a = 0,80",
            "Wzrost gospodarczy",
            "Wzrost gospodarczy dla a = 0,80"
        ),
    ]

    group_titles = {
        "K0 = 50": "Zmiana początkowego zasobu kapitału: K₀ = 50",
        "K0 = 200": "Zmiana początkowego zasobu kapitału: K₀ = 200",
        "alk0 = 5": (
            "Zmiana początkowego średniego okresu "
            "użytkowania kapitału: alk₀ = 5"
        ),
        "alk0 = 20": (
            "Zmiana początkowego średniego okresu "
            "użytkowania kapitału: alk₀ = 20"
        ),
        "a = 0,60": "Zmiana skłonności do konsumpcji: a = 0,60",
        "a = 0,80": "Zmiana skłonności do konsumpcji: a = 0,80",
    }

    current_group = None

    for chart_index in range(
        0,
        len(sensitivity_charts),
        2
    ):
        variant = sensitivity_charts[chart_index][0]

        if variant != current_group:
            st.markdown(
                f"### {group_titles[variant]}"
            )
            current_group = variant

        col1, col2 = st.columns(2)

        with col1:
            variant, column, title = sensitivity_charts[chart_index]

            st.plotly_chart(
                sensitivity_chart(
                    sensitivity_data[variant],
                    column,
                    title
                ),
                use_container_width=True,
                key=f"sensitivity_left_{chart_index}"
            )

        if chart_index + 1 < len(sensitivity_charts):
            with col2:
                variant, column, title = sensitivity_charts[
                    chart_index + 1
                ]

                st.plotly_chart(
                    sensitivity_chart(
                        sensitivity_data[variant],
                        column,
                        title
                    ),
                    use_container_width=True,
                    key=f"sensitivity_right_{chart_index}"
                )


with tab4:
    st.markdown("""
### Założenia modelu

- Symulacja przebiega w czasie dyskretnym z krokiem `dt = 0.01`; horyzont jest przeliczany na liczbę kroków.
- Parametry `E`, `ζ`, `R` i pozostałe parametry są stałe w czasie w ramach jednego scenariusza. Różne scenariusze mogą mieć różne wartości tych parametrów.
- `E` jest stałym w czasie współczynnikiem oddziaływania transformacji ekologicznej; w interfejsie przyjmuje wartości od `0.01` do `1.0`, a `E = 1` oznacza poziom referencyjny.
- `ζ` określa siłę wpływu `E` na produkcję przez `E^ζ`. Przy `E = 1` jego zmiana nie wpływa na wyniki, dlatego należy go interpretować razem z `E < 1`.
- `R` jest stałą marżą/kosztem kapitału. Wpływa na koszt użytkowania kapitału, `KLR` oraz wyznaczanie `α`.
- `γ₀` określa podstawową siłę dostosowania `alk`, a `γ_E` dodatkowo waży to dostosowanie zależnie od odchylenia środowiska od poziomu referencyjnego przez czynnik `(1 - E)`.
- W kodzie oba parametry działają ze znakiem minus: `[-γ₀ − γ_E · (1 − E)] · (KLR/KLR₀ − 1)`. Ich interpretacja wynika więc z tej konwencji znaków, a nie ze wzoru z dodatnim `γ₀`.
- Funkcja produkcji ma postać Cobba-Douglasa: `q = q₀ · Eᶻᵉᵗᵃ · (K/K₀)^α · (L/L₀)^β`, gdzie `β = 1 − α`.
- Populacja rośnie wykładniczo zgodnie z `Popₙ = Pop₀ · exp(n · tₙ)`. Praca utrzymuje stały początkowy wskaźnik zatrudnienia `L₀ / Pop₀`, dlatego `Lₙ = (L₀ / Pop₀) · Popₙ`.
- Początkowy udział kapitału wynika ze wzoru `α₀ = clip[K₀/q₀ · (1/alk₀ + R), 0.02, 0.98]`. Kolejne wartości `α` są aktualizowane analogicznie na podstawie poprzedniego okresu.
- `alk₀` jest wartością początkową. Dla `n > 0` najpierw wyznaczane jest `alkₙ` na podstawie znanego z poprzedniego kroku `KLRₙ₋₁`, a następnie z `alkₙ` obliczane jest `KLRₙ`.
- Inwestycja planowana wynika ze zmiany konsumpcji: `Iᵖₙ = max[b · (Cₙ − Cₙ₋₁), 0]`, a inwestycja faktyczna jest korektą zapewniającą równowagę `Pq · qₙ = Cₙ + Iₙ + Gₙ + Xₙ − Mₙ`.
- Kapitał zmienia się zgodnie z inwestycją pomniejszoną o zużycie `Kₙ/alkₙ`. Ujemny kapitał jest ograniczany do `10⁻⁸` wyłącznie dla stabilności obliczeń.
- Płaca jest równa krańcowemu produktowi pracy: `rwₙ = βₙ · qₙ / Lₙ`. Zysk uwzględnia przychód, koszt kapitału, zużycie kapitału i koszt pracy.
- Import zależy od `alk`, a eksport od relacji `KLR/KLR₀`. Parametry `η` i `κ` określają odpowiednie elastyczności.
- Zabezpieczenia `max(..., 10⁻¹²)` i `clip(...)` chronią przed dzieleniem przez zero oraz wartościami spoza stabilnego zakresu.
""")


with tab5:
    st.subheader("Wzory używane w modelu")
    st.caption(
        "Równania są zapisywane dla okresu n. Zmienne z indeksem n−1 "
        "pochodzą z poprzedniego kroku symulacji."
    )

    st.markdown("#### Warunki początkowe i parametry")

    st.latex(
        r"KLR_0 = \frac{K_0}{L_0} = "
        r"\frac{\alpha_0 rw_0}"
        r"{\left(\frac{1}{alk_0} + R\right)(1-\alpha_0)}"
    )

    st.latex(r"\beta_n = 1 - \alpha_n")
    st.latex(r"Pop_n = Pop_0 e^{n \cdot t_n}")
    st.latex(r"L_n = \frac{L_0}{Pop_0} \cdot Pop_n")

    st.markdown("#### Produkcja i technologia")

    st.latex(
        r"q_n = q_0 \cdot E^{\zeta} \cdot "
        r"\left(\frac{K_n}{K_0}\right)^{\alpha_n} \cdot "
        r"\left(\frac{L_n}{L_0}\right)^{\beta_n}"
    )

    st.latex(
        r"alk_n = alk_0 \cdot "
        r"e^{[-\gamma_0 - \gamma_E(1-E)]"
        r"\left(\frac{KLR_{n-1}}{KLR_0} - 1\right)}"
        r"\quad (n > 0)"
    )

    st.latex(r"alk_0 = alk_0")
    st.latex(r"KOR_n = \frac{K_n}{q_n}")
    st.latex(r"rw_n = \frac{\beta_n q_n}{L_n}")

    st.latex(
        r"KLR_n = \frac{K_n}{L_n} = "
        r"\frac{\alpha_n rw_n}"
        r"{\left(\frac{1}{alk_n} + R\right)\beta_n}"
    )

    st.latex(
        r"\alpha_n \text{ jest wyznaczane numerycznie z } "
        r"\alpha_n = KOR_n\left(\frac{1}{alk_n} + R\right),"
        r"\quad 0.02 \leq \alpha_n \leq 0.98"
    )

    st.markdown("#### Popyt, handel i równowaga rynkowa")

    st.latex(r"C_n = a \cdot P_q \cdot q_n")
    st.latex(r"G_n = GovSp \cdot P_q \cdot q_n")

    st.latex(
        r"M_n = m_0 \cdot P_q \cdot q_n \cdot "
        r"\left(\frac{alk_0}{alk_n}\right)^{\eta}"
    )

    st.latex(
        r"X_n = x_0 \cdot P_q \cdot q_n \cdot "
        r"\left(\frac{KLR_n}{KLR_0}\right)^{\kappa}"
    )

    st.latex(
        r"I_n^p = \max\left[b(C_n - C_{n-1}),\ 0\right], "
        r"\quad I_0^p = 0"
    )

    st.latex(
        r"Y_n^p = C_n + I_n^p + G_n + X_n - M_n"
    )

    st.latex(
        r"I_n = I_n^p + P_q q_n - Y_n^p"
    )

    st.latex(
        r"Y_n = C_n + I_n + G_n + X_n - M_n"
    )

    st.latex(r"P_q q_n = Y_n")

    st.markdown("#### Kapitał, zysk i wskaźniki")

    st.latex(
        r"K_{n+1} = \max\left[K_n + \Delta t"
        r"\left(\frac{I_n}{P_q} - \frac{K_n}{alk_n}\right),"
        r"\ 10^{-8}\right]"
    )

    st.latex(
        r"\pi_n = P_q q_n - P_k K_n"
        r"\left(\frac{1}{alk_n} + R\right)"
        r" - P_q L_n rw_n"
    )

    st.latex(
        r"u_n = 1 - \frac{L_n}{Pop_n}"
    )

    st.latex(
        r"g_{q,n}^{\mathrm{równanie}} = "
        r"\alpha_n g_{K,n} + \beta_n g_{L,n} + "
        r"KOR_n \frac{rw_n}{KLR_n} "
        r"\ln\left(\frac{KLR_n}{KLR_0}\right)"
        r"\left[\frac{\dot{KLR}_n}{KLR_n} "
        r"- \frac{\dot{rw}_n}{rw_n}\right]"
    )
