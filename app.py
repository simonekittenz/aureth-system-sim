import streamlit as st

st.set_page_config(page_title="Aureth and Bajurinda", layout="wide")

pages = st.navigation([
    st.Page("moons.py", title="Bayurinda moons", icon=":material/dark_mode:", default=True),
    st.Page("system.py", title="Aureth system", icon=":material/wb_sunny:"),
])
pages.run()
