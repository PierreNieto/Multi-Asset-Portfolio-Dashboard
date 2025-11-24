#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st

# Import des modules
from app.quantA.single_asset_page import run_single_asset_page
from app.portfolio.portfolio_page import run_portfolio_page

# ---------------------------
# MAIN DASHBOARD NAVIGATION
# ---------------------------

st.set_page_config(
    page_title="Quant Dashboard",
    layout="wide",
)

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a module",
    ["Single Asset (Quant A)", "Multi-Asset Portfolio (Quant B)"]
)

if page == "Single Asset (Quant A)":
    run_single_asset_page()

elif page == "Multi-Asset Portfolio (Quant B)":
    run_portfolio_page()
