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
server = app.server  # Needed for deployment

# Default Parameters
DEFAULT_MEAN = 1.0
DEFAULT_STD = 1.0
DEFAULT_POINTS = 100
DEFAULT_MULTIPLIER = 1.0
GRAPH_TYPES = ["Line", "Bar", "Scatter"]

# Function to Parse Uploaded File (CSV or Parquet)
def parse_file(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(decoded))
        else:
            raise ValueError("Unsupported file type")
        return df
    except Exception as e:
        print(f"Error parsing file: {e}")
        return None

# Function to Generate Graph Data
def generate_data(mean, std, points, multiplier):
    x_values = np.linspace(0, 10, points)
    y_values = np.sin(x_values) * multiplier + np.random.normal(mean, std, points)
    return x_values, y_values

# Function to Generate ML Model Predictions
def generate_ml_predictions():
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2.5 * X + np.random.randn(100, 1) * 2

    model = LinearRegression()
    model.fit(X, y)
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred = model.predict(X_test)

    return X.flatten(), y.flatten(), X_test.flatten(), y_pred.flatten()

# Function to Generate Deep Learning Training Loss
def generate_training_loss():
    epochs = np.arange(1, 51)
    loss = np.exp(-epochs / 10) + np.random.normal(0, 0.02, 50)
    return epochs, loss

# Navigation Bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Live Graph", href="/")),
        dbc.NavItem(dbc.NavLink("Cosine Graph", href="/graph2")),
        dbc.NavItem(dbc.NavLink("Random Data", href="/graph3")),
        dbc.NavItem(dbc.NavLink("ML Model Predictions", href="/ml")),
        dbc.NavItem(dbc.NavLink("Deep Learning Training Loss", href="/dl")),
        dbc.NavItem(dbc.NavLink("EEG Classes", href="/classes")),
    ],
    brand="Elektra: Harmful Brain Activity Classification",
    brand_href="/",
    color="primary",
    dark=True,
    className="mb-4"
)

# Sidebar Controls
controls = html.Div([
    html.Label("Mean:"),
    dcc.Slider(id="mean-slider", min=0, max=5, step=0.1, value=DEFAULT_MEAN, marks={i: str(i) for i in range(6)}),
    
    html.Label("Std Dev:"),
    dcc.Slider(id="std-slider", min=0.1, max=3, step=0.1, value=DEFAULT_STD, marks={i: str(i) for i in range(1, 4)}),
    
    html.Label("Data Points:"),
    dcc.Slider(id="points-slider", min=50, max=500, step=50, value=DEFAULT_POINTS, marks={i: str(i) for i in range(50, 501, 50)}),
    
    html.Label("Multiplier:"),
    dcc.Slider(id="multiplier-slider", min=0.5, max=3, step=0.1, value=DEFAULT_MULTIPLIER, marks={i: str(i) for i in range(1, 4)}),

    html.Label("Graph Type:"),
    dcc.Dropdown(id="graph-type", options=[{"label": t, "value": t} for t in GRAPH_TYPES], value="Line", clearable=False),

    html.Br(),
    dcc.Upload(
        id="upload-data",
        children=html.Button("Upload CSV or Parquet File"),
        multiple=False,
        style={"marginTop": "10px"}
    ),

    html.Div(id="csv-message", style={"color": "green", "fontWeight": "bold", "marginTop": "10px"}),

], style={"width": "25%", "float": "left", "padding": "20px"})

# Graph Display
graph_display = html.Div([
    dcc.Graph(id="main-graph"),
], style={"width": "70%", "float": "right"})

# Define Page Layouts
page1_layout = html.Div([controls, graph_display])
page2_layout = html.Div([html.H3("Graph 2: Cosine Wave"), dcc.Graph(id="graph2")])
page3_layout = html.Div([html.H3("Graph 3: Random Data"), dcc.Graph(id="graph3")])

# NEW: EEG Classes Layout
classes_layout = html.Div([
    html.H3("EEG Classification Labels"),
    dbc.ListGroup([
        dbc.ListGroupItem("Seizure"),
        dbc.ListGroupItem("LRDA (Lateralized Rhythmic Delta Activity)"),
        dbc.ListGroupItem("GRDA (Generalized Rhythmic Delta Activity)"),
        dbc.ListGroupItem("LPD (Lateralized Periodic Discharges)"),
        dbc.ListGroupItem("GPD (Generalized Periodic Discharges)"),
        dbc.ListGroupItem("Other")
    ], style={"maxWidth": "600px", "marginTop": "20px"})
], style={"padding": "20px"})

# App Layout
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    navbar,
    html.Div(id="page-content")
])

# Callback to Switch Between Pages
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/graph2":
        x = np.linspace(0, 10, 100)
        y = np.cos(x)
        figure = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines", name="Cosine Graph")])
        return html.Div([html.H3("Graph 2: Cosine Wave"), dcc.Graph(figure=figure)])

    elif pathname == "/graph3":
        x = np.linspace(0, 10, 100)
        y = np.random.rand(100)
        figure = go.Figure(data=[go.Scatter(x=x, y=y, mode="markers", name="Random Data")])
        return html.Div([html.H3("Graph 3: Random Data"), dcc.Graph(figure=figure)])

    elif pathname == "/classes":
        return classes_layout

    else:
        return page1_layout  # Default page is Live Graph page

# Callback to Update Graph (Now Updates When Sliders are Released)
@app.callback(
    [Output("main-graph", "figure"),
     Output("csv-message", "children")],
    [Input("upload-data", "contents"),
     Input("mean-slider", "value"),
     Input("std-slider", "value"),
     Input("points-slider", "value"),
     Input("multiplier-slider", "value"),
     Input("graph-type", "value")],
    State("upload-data", "filename"),
    prevent_initial_call=False
)
def update_graph(csv_contents, mean, std, points, multiplier, graph_type, filename):
    message = ""

    # If no file uploaded, use generated data
    if not csv_contents:
        x_values, y_values = generate_data(mean, std, points, multiplier)
        message = "⚠️ No file uploaded. Using generated data."
    else:
        df = parse_file(csv_contents, filename)
        if df is not None:
            x_values = df.iloc[:, 0]
            y_values = df.iloc[:, 1]
            message = f"✅ File Loaded: {filename}"
        else:
            x_values, y_values = generate_data(mean, std, points, multiplier)
            message = "⚠️ Error loading file. Using generated data."

    # Select graph type
    if graph_type == "Line":
        trace = go.Scatter(x=x_values, y=y_values, mode="lines", name="Line Graph")
    elif graph_type == "Bar":
        trace = go.Bar(x=x_values, y=y_values, name="Bar Graph")
    elif graph_type == "Scatter":
        trace = go.Scatter(x=x_values, y=y_values, mode="markers", name="Scatter Plot")

    # Create and return figure
    figure = go.Figure(data=[trace])
    return figure, message

# Run App
if __name__ == "__main__":
    app.run_server(debug=True)
