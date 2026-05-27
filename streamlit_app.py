import math
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Economic Growth Model", layout="wide")

st.title("📈 Economic Growth Model")
st.markdown(
    "This app simulates a stylized economic growth model with time-varying capital share, labor share, and capital accumulation."
)

with st.sidebar:
    st.header("Model parameters")
    q0 = st.number_input("Initial output \(q_0\)", value=100.0, min_value=0.1)
    K0 = st.number_input("Initial capital \(K_0\)", value=120.0, min_value=0.1)
    L0 = st.number_input("Initial labor \(L_0\)", value=80.0, min_value=0.1)
    alk0 = st.number_input("Initial capital lifetime \(alk_0\)", value=20.0, min_value=0.1)
    KLR0 = st.number_input("Initial capital-labor ratio \(KLR_0\)", value=1.5, min_value=0.01)
    R = st.number_input("Parameter \(R\)", value=0.02)
    gamma = st.number_input("Adjustment speed \(\gamma\)", value=0.12, min_value=0.0)
    Pop0 = st.number_input("Initial population \(Pop_0\)", value=100.0, min_value=0.1)
    n = st.number_input("Population growth rate \(n\)", value=0.01)
    labor_share = st.slider("Employment rate \(\lambda\)", min_value=0.1, max_value=1.0, value=0.8, step=0.01)
    GovSp = st.slider("Government spending share \(GovSp\)", min_value=0.0, max_value=0.5, value=0.2, step=0.01)
    s = st.slider("Investment share \(s\)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    m0 = st.number_input("Import intensity \(m_0\)", value=0.05, min_value=0.0)
    eta = st.number_input("Import elasticity \(\eta\)", value=0.5, min_value=0.0)
    x0 = st.number_input("Export intensity \(x_0\)", value=0.05, min_value=0.0)
    kappa = st.number_input("Export elasticity \(\kappa\)", value=0.5, min_value=0.0)
    T = st.slider("Simulation horizon (years)", min_value=10, max_value=200, value=80, step=5)
    dt = st.select_slider("Time step \(\Delta t\)", options=[0.1, 0.25, 0.5, 1.0], value=0.5)

st.markdown("---")

st.header("Simulation results")

# Time grid
steps = int(T / dt) + 1
times = [i * dt for i in range(steps)]

# Initialize state variables
K = K0
q = q0

# Storage
Pop = []
L = []
KLR = []
alk = []
KOR = []
alpha = []
beta = []
Y = []
G = []
I = []
C = []
M = []
X = []
rw = []
u_rate = []
K_series = []
q_series = []
dalpha_dt = []

for t in times:
    pop_t = Pop0 * math.exp(n * t)
    labor_t = labor_share * pop_t
    k_l_ratio = K / labor_t
    adjustment_lifetime = alk0 * math.exp(-gamma * (k_l_ratio / KLR0 - 1.0))

    # Solve the implicit output equation q(t) using fixed-point iteration
    q_t = max(1e-8, q)
    for _ in range(50):
        k_or_guess = K / q_t if q_t != 0 else 0.0
        alpha_guess = k_or_guess * ((1.0 / adjustment_lifetime) + R)
        alpha_guess = max(1e-6, min(alpha_guess, 1.0 - 1e-6))
        beta_guess = 1.0 - alpha_guess
        q_next = q0 * ((K / K0) ** alpha_guess) * ((labor_t / L0) ** beta_guess)
        if q_next <= 0:
            q_next = 1e-8
        if abs(q_next - q_t) < 1e-8:
            q_t = q_next
            break
        q_t = q_next

    k_or = K / q_t if q_t != 0 else 0.0
    alpha_t = k_or * ((1.0 / adjustment_lifetime) + R)
    alpha_t = max(1e-6, min(alpha_t, 1.0 - 1e-6))
    beta_t = 1.0 - alpha_t
    y_t = q_t
    g_t = GovSp * y_t
    i_t = s * y_t
    m_t = m0 * y_t * ((alk0 / adjustment_lifetime) ** eta)
    x_t = x0 * y_t * ((k_l_ratio / KLR0) ** kappa)
    c_t = y_t - g_t - i_t - x_t + m_t
    rw_t = beta_t * y_t / labor_t if labor_t != 0 else 0.0
    u_t = 1.0 - labor_t / pop_t if pop_t != 0 else 0.0

    if not alpha:
        dalk_dt = 0.0
        dKOR_dt = 0.0
    else:
        dalk_dt = (adjustment_lifetime - alk[-1]) / dt
        dKOR_dt = (k_or - KOR[-1]) / dt

    dalpha = (
        -k_or / (adjustment_lifetime ** 2) * dalk_dt
        + ((1.0 / adjustment_lifetime) + R) * dKOR_dt
    )

    Pop.append(pop_t)
    L.append(labor_t)
    KLR.append(k_l_ratio)
    alk.append(adjustment_lifetime)
    KOR.append(k_or)
    alpha.append(alpha_t)
    beta.append(beta_t)
    Y.append(y_t)
    G.append(g_t)
    I.append(i_t)
    C.append(c_t)
    M.append(m_t)
    X.append(x_t)
    rw.append(rw_t)
    u_rate.append(u_t)
    dalpha_dt.append(dalpha)
    K_series.append(K)
    q_series.append(q_t)

    # Capital accumulation using Euler update
    K_dot = i_t - (K / adjustment_lifetime)
    K = max(0.0, K + dt * K_dot)
    q = q_t

col1, col2 = st.columns(2)
with col1:
    st.subheader("Key end values")
    st.metric("Output Y(T)", f"{Y[-1]:,.2f}")
    st.metric("Capital K(T)", f"{K_series[-1]:,.2f}")
    st.metric("Labor share β(T)", f"{beta[-1]:.4f}")

with col2:
    st.subheader("Additional end values")
    st.metric("Capital share α(T)", f"{alpha[-1]:.4f}")
    st.metric("Capital-labor ratio K/L(T)", f"{KLR[-1]:.4f}")
    st.metric("Real wage rw(T)", f"{rw[-1]:,.2f}")

st.markdown("---")

# Build dataframes for charting with x-axis domain starting at 0
chart_df = pd.DataFrame(
    {
        "time": times,
        "Output Y": Y,
        "Capital K": K_series,
        "Labor L": L,
        "Capital share α": alpha,
        "Labor share β": beta,
        "Capital-labor ratio K/L": KLR,
        "Capital lifetime alk": alk,
        "Government spending G": G,
        "Investment I": I,
        "Consumption C": C,
        "Exports X": X,
        "Imports M": M,
        "dα/dt": dalpha_dt,
        "Population Pop": Pop,
        "Employment rate u": u_rate,
        "Real wage rw": rw,
    }
)

def alt_line_chart(column, title):
    df = chart_df[["time", column]].rename(columns={column: "value"})
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(
                "time:Q",
                title="Time",
                scale=alt.Scale(domainMin=0, domainMax=times[-1], nice=False, clamp=True),
            ),
            y=alt.Y("value:Q", title=title),
            tooltip=["time:Q", "value:Q"],
        )
        .properties(title=title)
        
    )


def render_series_section(title, columns):
    st.subheader(title)
    for i in range(0, len(columns), 3):
        row = st.columns(3)
        for chart_column, container in zip(columns[i:i+3], row):
            with container:
                st.altair_chart(alt_line_chart(chart_column, chart_column), use_container_width=True)

render_series_section("Output, Capital, Labor", ["Output Y", "Capital K", "Labor L"])
render_series_section("Distribution and Ratios", ["Capital share α", "Labor share β", "Capital-labor ratio K/L"])
render_series_section("Dynamics", ["dα/dt", "Capital lifetime alk", "Government spending G"])
render_series_section("Financial Aggregates", ["Investment I", "Consumption C", "Exports X"])
render_series_section("External Sector", ["Imports M"])
render_series_section("Population and Real Wage", ["Population Pop", "Employment rate u", "Real wage rw"])

st.markdown("---")

st.write("### Model equations")
st.latex(r"q(t) = q_0 \left( \frac{K(t)}{K_0} \right)^{\alpha(t)} \left( \frac{L(t)}{L_0} \right)^{\beta(t)}")
st.latex(r"KLR(t) = \frac{K(t)}{L(t)}")
st.latex(r"alk(t) = alk_0 \exp\left[-\gamma \left( \frac{KLR(t)}{KLR_0} - 1 \right)\right] ")
st.latex(r"\alpha(t) = KOR(t) \left( \frac{1}{alk(t)} + R \right)")
st.latex(r"\beta(t) = 1 - \alpha(t)")
st.latex(r"KOR(t) = \frac{K(t)}{q(t)}")
st.latex(r"\frac{d\alpha}{dt} = -\frac{KOR(t)}{[alk(t)]^2} \frac{d(alk(t))}{dt} + \left(\frac{1}{alk(t)} + R\right) \frac{dKOR(t)}{dt}")
st.latex(r"rw(t) = \beta(t) \frac{q(t)}{L(t)}")
st.latex(r"Pop(t) = Pop_0 e^{nt}")
st.latex(r"L(t) = \lambda Pop(t)")
st.latex(r"u(t) = 1 - \frac{L(t)}{Pop(t)}")
st.latex(r"G(t) = GovSp \cdot Y(t)")
st.latex(r"Y(t) = q(t) = C(t) + I(t) + G(t) + X(t) - M(t)")
st.latex(r"\dot K(t) = I(t) - \frac{K(t)}{alk(t)}")
st.latex(r"X(t) = x_0 * Y(t) * (\frac{KLR(t)}{KLR_0})^\kappa")
st.latex(r"M(t) = m_0 * Y(t) * \frac{alk_0}{alk(t)}^\eta")

st.caption("Note: The model uses a simple Euler integration for capital accumulation over time.")
