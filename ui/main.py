import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
import io
from sklearn.linear_model import LinearRegression

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server  

# Run App
if __name__ == "__main__":
    app.run_server(debug=True)
