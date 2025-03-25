from dash import Dash, html

app = Dash(__name__)

app.layout = html.Div(children=[
  html.H1("Hello, Dash!"),
  html.H2("This is a heading"),
  html.P("This is a paragraph"),
])

if __name__ == '__main__':
  app.run(debug=True)
