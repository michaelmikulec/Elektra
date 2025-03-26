from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import io
import base64

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1("Elektra"),
    html.H2("Harmful Brain Activity Classification"),
    
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload CSV or Parquet'),
        multiple=False
    ),
    
    html.Div(id='output-data-upload')
])

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_file(contents, filename):
    if contents is None:
        return "No file uploaded yet."

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(decoded))
        else:
            return "Unsupported file format. Please upload a CSV or Parquet file."

        return html.Div([
            html.H5(f"Loaded {filename}"),
            html.Pre(df.head().to_string(index=False))
        ])

    except Exception as e:
        return f"Error loading file: {e}"

if __name__ == '__main__':
    app.run(debug=True)
