
from dash import Dash, dcc, html, Input, Output, State

import pandas as pd              # for tabular output

from select_worker import calculate_best_choise


data = pd.read_csv(r"C:\Users\IlyaY\Desktop\לימודים\תשפג\א\אלגו\project\data\DAC_NationalDownloadableFile_upd_new.csv")  # (r"C:\Users\IlyaY\Desktop\לימודים\תשפג\א\סייבר\project\data\less_data.csv")
data["years_experience"] = 2022 - data.Grd_yr
print(data.columns)
print(data.years_experience)


app = Dash(__name__)


app.layout = html.Div([
    # Title
    html.H1('Welcome to stuff selecting App!', style={
        'textAlign': 'center',
        'color': '#444444',
        'fontSize': 24
    }),
    # Text
    html.P(['Please select gender of the worker that you want to hire', html.Br(), ' Also , select the specefication that you need to hire to.'
                                                                                ,html.Br(),
           'The program will access the biggest dataset avaliable and you will recieve the best option ! '], style={
        'textAlign': 'center',
        'color': '#666666',
        'fontSize': 16
    }),

    dcc.Dropdown(data["gndr"].unique(), id='gender_filter', style={'width': '50%', 'textAlign': 'center',
        'color': '#666666',
        'fontSize': 16,},value='M'),
    dcc.Dropdown(data["pri_spec"].unique(), id='spec_filter', style={'width': '50%', 'textAlign': 'center',
        'color': '#666666',
        'fontSize': 16}, value='CHIROPRACTIC'),
    html.Div(id='out', style={'width': '50%', 'textAlign': 'center',
        'color': '#666666',
        'fontSize': 16}),

],
style={
        'textAlign': 'center',
        'color': '#666666',
        'fontSize': 16
    }
)

@app.callback(
    Output(component_id='out', component_property='children'),
    [Input('gender_filter', 'value'),
     Input('spec_filter', 'value')]
)

def update_output(gender_filter, spec_filter):
    try:
        return f"{calculate_best_choise(gender_filter, spec_filter)}"
    except:
        return "No such query"





if __name__ == '__main__':
    app.run_server(debug=True)










