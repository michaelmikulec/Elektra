from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import io
import base64
import random

# Simulate model loading (replace with actual model)
MODEL_ACCURACY = 0.89  # e.g., 89% test accuracy

EEG_CATEGORIES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1("Elektra"),
    html.H2("Harmful Brain Activity Classification"),

    html.Div([
        html.P("Classifies input EEG datasets into one of 6 categories:"),
        html.Ul([html.Li(f"{i+1}. {cat}") for i, cat in enumerate(EEG_CATEGORIES)])
    ]),

    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload CSV or Parquet'),
        multiple=False
    ),

    html.Div(id='output-data-upload'),
    
    html.H3("Display Elements"),
    html.Ul([
        html.Li(id='model-accuracy'),
        html.Li(id='predicted-class'),
        html.Li(id='prediction-confidence')
    ])
])

@app.callback(
    Output('output-data-upload', 'children'),
    Output('model-accuracy', 'children'),
    Output('predicted-class', 'children'),
    Output('prediction-confidence', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_and_infer(contents, filename):
    if contents is None:
        return "No file uploaded yet.", "", "", ""

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(decoded))
        else:
            return "Unsupported file format. Please upload a CSV or Parquet file.", "", "", ""

        # Placeholder for actual inference
        predicted_index = random.randint(0, 5)
        predicted_class = EEG_CATEGORIES[predicted_index]
        confidence = round(random.uniform(0.6, 0.99), 2)

        return (
            html.Div([
                html.H5(f"Loaded {filename}"),
                html.Pre(df.head().to_string(index=False))
            ]),
            f"Model Accuracy: {MODEL_ACCURACY * 100:.2f}%",
            f"Inferred Classification: {predicted_class}",
            f"Confidence: {confidence * 100:.2f}%"
        )

    except Exception as e:
        return f"Error loading file: {e}", "", "", ""

if __name__ == '__main__':
    app.run(debug=True)
