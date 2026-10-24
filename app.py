import streamlit as st

st.set_page_config(page_title="Aureth and Bajurinda", layout="wide")


def reload_data():
    """Re-read everything in the data folder, keeping the settings and any unsaved constellation edits."""
    from bajurinda import nightsky
    from bajurinda.constellations import clone, load_constellations
    from bajurinda.starfield import load_faint_settings, load_sky

    ss = st.session_state
    from bajurinda.world import load_world
    try:                                    # check the files are readable before throwing away what is loaded
        load_sky.__wrapped__()
        load_world()
        load_faint_settings()
        fresh = load_constellations()
    except Exception as err:                # noqa: BLE001 - show whatever went wrong in the data file
        ss["_reload_err"] = f"{type(err).__name__}: {err}"
        st.toast("Reload failed: a data file has a mistake. Fix it and press reload again.", icon="⚠️")
        return
    ss.pop("_reload_err", None)
    load_sky.cache_clear()
    nightsky._FAINT_CACHE.clear()
    st.cache_data.clear()
    kept = False
    if "_con_work" in ss:
        if ss["_con_work"] == ss.get("_con_disk"):
            ss["_con_work"] = clone(fresh)
        else:
            kept = True                     # unsaved constellation edits win; don't throw them away
        ss["_con_disk"] = clone(fresh)
    st.toast("Data reloaded." + (" Your unsaved constellation edits were kept." if kept else ""), icon="🔄")


pages = st.navigation([
    st.Page("planet_view.py", title="Planet view", icon=":material/public:", default=True),
    st.Page("system.py", title="Aureth system", icon=":material/wb_sunny:"),
])
with st.sidebar:
    st.button("Reload data files", on_click=reload_data, icon=":material/refresh:", key="reload_data", width="stretch",
              help="Re-reads data/aureth.yaml and data/constellations.yaml after you edit them by hand, and clears the saved calculations. "
                   "Your sidebar settings stay as they are.")
    if st.session_state.get("_reload_err"):
        st.error("A data file has a mistake and can't be read. Fix it, then press Reload again:\n\n" + st.session_state["_reload_err"])
pages.run()
