#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st

from app.single_asset.single_asset_page import run_single_asset_page
from app.portfolio.portfolio_page import run_portfolio_page

# ---------------------------
# MAIN DASHBOARD NAVIGATION
# ---------------------------

st.set_page_config(
    page_title="Quant Dashboard",
    layout="wide",
)

st.sidebar.title("Navigation")

menu_options = ["Multi-Asset Portfolio (Quant B)", "Single Asset (Quant A)"]

page = st.sidebar.selectbox(
    "Choose a module",
    menu_options,
    index=0
)

if page == "Multi-Asset Portfolio (Quant B)":
    run_portfolio_page()

elif page == "Single Asset (Quant A)":
    run_single_asset_page()