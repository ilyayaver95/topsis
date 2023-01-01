import numpy as np
import pandas as pd
from collections import Counter
import dash
import os
from dash import Dash, dcc, html, Input, Output, State
import base64
import numpy as np               # for linear algebra
import pandas as pd              # for tabular output
from scipy.stats import rankdata # for ranking the candidates
from select_worker import calculate_best_choise

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\IlyaY\Desktop\לימודים\תשפג\א\סייבר\project\data\less_data.csv")
data["years_experience"] = 2022 - data.Grd_yr
print(data.columns)
print(data.years_experience)

# # experience frequency
# plt.figure(figsize=(14,6))
# plt.hist(data.years_experience, edgecolor="k",density=True)
# plt.xlabel("years_experience")
# plt.ylabel("Frequency")
# plt.show()

# # experience frequency male\female
# plt.figure(figsize=(14,6))
# plt.hist(data[data["gndr"]=="F"]["years_experience"], edgecolor="k",density=True, alpha=0.7, label = "Female")
# plt.hist(data[data["gndr"]=="M"]["years_experience"], edgecolor="k",density=True, alpha=0.7, label = "Male")
# plt.xlabel("years_experience")
# plt.ylabel("Frequency")
# plt.legend()
# plt.show()

# # Cred by frequency
# data['Cred'].value_counts().plot(kind='bar')
# plt.show()

# # pri_spec by frequency
# data['pri_spec'].value_counts().plot(kind='bar')
# plt.show()

# # Med_sch by frequency
# data = data[data["Med_sch"] != 'OTHER']
# data['Med_sch'].value_counts().plot(kind='bar')
# plt.show()



# plan:
# 1) shoe distrebution of different features: experience, education, epxert, sex
# 2) create ideal position

app = Dash(__name__)
# app.layout = html.Div([
#     dcc.Dropdown(data["gndr"].unique(), id='gndr_dropdown'),
#     dcc.Dropdown(data["pri_spec"].unique(), id='pri_spec_dropdown'),
#     html.Div(id='pandas-output-container-2'),
#     html.Br(),
#     html.Br(),
#     html.Div(id='result')
#
# ])


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
# , style = {
#     'width': '80%',
#     'marginLeft': 'auto',
#     'marginRight': 'auto',
#     'paddingTop': 20,
#     'paddingBottom': 20}
)


# @app.callback(
#     Output('pandas-output-container-2', 'children'),
#     Input('gndr_dropdown', 'value'),
#     Input('pri_spec_dropdown', 'value')
# )
@app.callback(
    Output(component_id='out', component_property='children'),
    [Input('gender_filter', 'value'),
     Input('spec_filter', 'value')]
)

def update_output(gender_filter, spec_filter):
    # return f'You have selected {value1} , {value2}'
    # path = r"C:\Users\IlyaY\Desktop\לימודים\תשפג\א\אלגו\project\data\less_data.csv"
    # gender_filter = 'M'
    # spec_filter = 'CHIROPRACTIC'
    try:
        return f"{calculate_best_choise(gender_filter, spec_filter)}"
    except:
        return "No such query"





if __name__ == '__main__':
    app.run_server(debug=True)

#  todo: fix refreshing

















# topsis example:


# def rank_according_to(data):
#     ranks = rankdata(data).astype(int)
#     ranks -= 1
#     return candidates[ranks][::-1]
#
#
# attributes = np.array(["style", "reliability", "fuel_consumption", "price"])
# candidates = np.array(["honda", "ford", "mazda", "subaru"])
# raw_data = np.array([
#     [7, 9,  9,  8],
#     [8, 7,  8,  7],
#     [9, 6,  8,  9],
#     [6, 7,  8, 6]
# ])
#
# weights = np.array([0.1, 0.4, 0.3, 0.2])
#
# # The indices of the attributes (zero-based) that are considered beneficial.
# # Those indices not mentioned are assumed to be cost attributes.
# benefit_attributes = set([0, 1, 2, 3])
#
# # Display the raw data we have
# d = pd.DataFrame(data=raw_data, index=candidates, columns=attributes)
# print(d)
# print("")
#
# # Step 1 - Normalizing the ratings
# m = len(raw_data)
# n = len(attributes)
# divisors = np.empty(n)
# for j in range(n):
#     column = raw_data[:,j]
#     divisors[j] = np.sqrt(column @ column)
#
# raw_data = raw_data / divisors
#
# columns = ["X_{%d}" % j for j in range(n)]
# dd = pd.DataFrame(data=raw_data, index=candidates, columns=columns)
# print(dd)
# print("")
#
# # Step 2 - Calculating the Weighted Normalized Ratings
#
# raw_data *= weights
# ddd = pd.DataFrame(data=raw_data, index=candidates, columns=columns)
# print(ddd)
# print("")
#
# # Step 3 - Identifying PIS ( A∗ ) and NIS ( A− )
# a_pos = np.zeros(n)
# a_neg = np.zeros(n)
# for j in range(n):
#     column = raw_data[:, j]
#     max_val = np.max(column)
#     min_val = np.min(column)
#
#     # See if we want to maximize benefit or minimize cost (for PIS)
#     if j in benefit_attributes:
#         a_pos[j] = max_val
#         a_neg[j] = min_val
#     else:
#         a_pos[j] = min_val
#         a_neg[j] = max_val
#
# dddd = pd.DataFrame(data=[a_pos, a_neg], index=["A^*", "A^-"], columns=columns)
# print(dddd)
# print(" ")
#
# # Step 4 and 5 - Calculating Separation Measures and Similarities to PIS
#
# sp = np.zeros(m)
# sn = np.zeros(m)
# cs = np.zeros(m)
#
# for i in range(m):
#     diff_pos = raw_data[i] - a_pos
#     diff_neg = raw_data[i] - a_neg
#     sp[i] = np.sqrt(diff_pos @ diff_pos)
#     sn[i] = np.sqrt(diff_neg @ diff_neg)
#     cs[i] = sn[i] / (sp[i] + sn[i])
#
# ddddd = pd.DataFrame(data=zip(sp, sn, cs), index=candidates, columns=["S^*", "S^-", "C^*"])
# print(ddddd)
# print(" ")
#
# # Step 6 - Ranking the candidates/alternatives
# cs_order = rank_according_to(cs)
# sp_order = rank_according_to(sp)
# sn_order = rank_according_to(sn)
#
# dddddd = pd.DataFrame(data=zip(cs_order, sp_order, sn_order), index=range(1, m + 1), columns=["C^*", "S^*", "S^-"])
# print(dddddd)
# print(" ")
#
# print("The best candidate/alternative according to C* is " + cs_order[0])
# print("The preferences in descending order are " + ", ".join(cs_order) + ".")
