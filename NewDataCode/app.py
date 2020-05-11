import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from functions import get_data
import numpy as np

ColName = 'action_taken'
df1 = get_data(ColName,nr = 100000)
action_options = df1.columns
app = dash.Dash()

app.layout = html.Div([
    html.H2("HMDA Loan Prediction Project"),
    html.Div(
        [
            dcc.Dropdown(
                id="action",
                options=[{
                    'label': i,
                    'value': i
                } for i in action_options],
                value='loan_purpose'),
        ],
        style={'width': '25%',
               'display': 'inline-block'}),
    dcc.Graph(id='funnel-graph',figure = {}),
])

@app.callback(
    dash.dependencies.Output('funnel-graph', 'figure'),
    [dash.dependencies.Input('action', 'value')])

def update_graph(action):
    df = data.groupby(action).count()['action_taken']
    trace1 = go.Bar(x = [i for i in range(1,np.size(df.unique())+1)],y = df)
    return {
    'data': [trace1],
    'layout':
    go.Layout(
        title='Showing the count of number of occurences of values of column {} in the dataset'.format(action),
        barmode='stack')
    }

if __name__ == "__main__":
    ColName = 'action_taken'
    data = get_data(ColName,nr = 100000)
    app.run_server(debug=True)