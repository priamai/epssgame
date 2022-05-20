import json

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import requests
import pandas as pd
import numpy as np
import nvdlib
from dash.exceptions import PreventUpdate
import random

NVD_API_KEY = 'aae18486-d625-4b79-a2a1-883f59f3b7d6'

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

ticker = dcc.Interval(
            id='game-interval',
            interval=1*1000,
            disabled = True,
            n_intervals=0
        )

card_hover = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Hover events", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",id='hover-data'
                ),
                dbc.Button("Hide", color="primary"),
            ]
        ),
    ],
)

card_click = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Click events", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",id='click-data'
                ),
                dbc.Button("Hide", color="primary"),
            ]
        ),
    ],
)

card_select = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Hover events", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",id='selected-data'
                ),
                dbc.Button("Hide", color="primary"),
            ]
        ),
    ],
)

card_relayout = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Hover events", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",id='relayout-data'
                ),
                dbc.Button("Hide", color="primary"),
            ]
        ),
    ],
)

range_top = html.Div(
    [
        dbc.Label("Top CVE", html_for="slider",id='top-label'),
        dcc.Slider(id="top", min=10, max=100, step=10, value=1),
    ],
    className="mb-3",
)

slider_history = html.Div(
    [
        dbc.Label("Replay last 1 day", html_for="slider",id='replay-label'),
        dcc.Slider(id="replay", min=0, max=7, step=1, value=1),
    ],
    className="mb-3",
)

slider_set = html.Div(
    [
        dbc.Label("Game duration in days", html_for="slider"),
        dcc.Slider(id="duration", min=1, max=30, step=5, value=6),
    ],
    className="mb-3",
)

slider_time = html.Div(
    [
        dbc.Label("Game time", html_for="timer"),
        dbc.Progress(label="0", value=0,id='timer'),
    ],
    className="mb-3",
)

card_patched = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Patched", className="card-title"),
                html.P(
                    "The vulns you patched are here",
                    className="card-text"
                ),
            ],id='card-body-patched'
        ),
    ])


card_incidents = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Incidents", className="card-title"),
                html.P(
                    "Exploitation attacks are here",
                    className="card-text",id='text-incidents'
                ),
            ],id='card-body-attacks'
        ),
    ],
)

card_score= dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Your score", className="card-title"),
                html.P(
                    "Your score is here",
                    className="card-text",id='text-score'
                ),
            ],id='card-body-score'
        ),
    ],
)

loading_spinner = dbc.Spinner(html.Div(id="loading-spinner"))

app.layout = dbc.Container(
    [
        html.H2("EPSS Game v0.1", className="bg-primary text-white p-1"),
        #html.Hr(),
        dcc.Store(id='data-patched'),
        dcc.Store(id='data-attacks'),
        dcc.Store(id='data-epss'),
        ticker,
        dbc.Row(dbc.Col(dcc.Graph(id='basic-interactions'))),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(slider_history),
                dbc.Col(slider_set),
                dbc.Col(slider_time),
                dbc.Col(range_top),
            ],
            className="mt-4",
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Button("Start Game", id='game-button',color="primary")),
                dbc.Col(loading_spinner),
                dbc.Col(dbc.Button("Reset Game", id='reset-button',color="primary"))
            ], className="mt-2",
            ),
        dbc.Row(
                [dbc.Col(card_patched),
                dbc.Col(card_incidents),
                dbc.Col(card_score)], className="mt-2",),
        dbc.Row(
            [
                dbc.Col(card_hover),
                dbc.Col(card_click),
                dbc.Col(card_select),
                dbc.Col(card_relayout)
            ],
            className="mt-4",
        ),
    ],
    fluid=True
    )

import datetime
import dash

@app.callback(
    Output('game-button', 'children'),
    Output('game-interval', 'disabled'),
    Output('timer', 'label'),
    Output('timer', 'value'),
    Output('game-interval', 'n_intervals'),
    Input('game-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    State('game-button','children'),
    Input('game-interval', 'n_intervals'),
    State('game-interval', 'disabled'),
    State('duration','value')
    )
def display(game_button_clicks,game_reset,game_button_state,game_timer,game_timer_disabled,game_duration):
    ctx = dash.callback_context

    if not ctx.triggered:
        app.logger.info('No clicks yet')
    else:
        game_button_next = game_button_state
        game_disabled_next = game_timer_disabled
        game_bar_next = game_timer
        game_timer_next = game_timer
        for trigger in ctx.triggered:
            if trigger['prop_id'] == 'game-button.n_clicks':
                app.logger.info(f'Game button {trigger["value"]}')

                if game_button_state == 'Start Game': 
                    days_str = f'{game_timer} days'
                    game_button_next = 'Stop Game'
                    game_disabled_next = False

                elif game_button_state == 'Stop Game': 
                    days_str = f'{game_timer} days'
                    game_button_next = 'Start Game'
                    game_disabled_next = True

            elif trigger['prop_id'] == 'game-interval.n_intervals':
                days_str = f'{game_timer} days'
                app.logger.info(f'Game interval {trigger["value"]}')

            elif trigger['prop_id'] == 'reset-button.n_clicks':
                if game_reset > 0:
                    days_str = ''
                    game_bar_next = 0
                    game_timer_next = 0
                    app.logger.info(f'Reset Game {trigger["value"]}')

        ctx_msg = json.dumps({
            'states': ctx.states,
            'triggered': ctx.triggered,
            'inputs': ctx.inputs
        }, indent=2)

        app.logger.debug(ctx_msg)

        if game_timer >= game_duration:
            game_disabled_next = True

        return game_button_next,game_disabled_next,days_str,game_timer_next,game_bar_next

    raise PreventUpdate

@app.callback(
    Output('click-data', 'children'),
    Output('data-patched', 'data'),
    Input('basic-interactions', 'clickData'),
    State('data-patched', 'data'),
    State('game-interval', 'n_intervals'),
    State('game-interval', 'disabled'),
    Input('reset-button', 'n_clicks'))
def click_cve(clickData,patched_state,game_interval,game_disabled,reset_button):

    ctx = dash.callback_context

    if clickData is None:
        raise PreventUpdate
    elif game_interval <= 0:
        raise PreventUpdate
    elif game_disabled == True:
        raise PreventUpdate
    else:
        if patched_state is None: patched_state = {}
        
        for trigger in ctx.triggered:
            if trigger['prop_id'] == 'reset-button.n_clicks':
                app.logger.info(f'Reset button {trigger["value"]}')
                patched_state = {}
                break
            elif trigger['prop_id'] == 'basic-interactions.clickData':
                app.logger.info(f"Total patch actions {len(patched_state)}")
                for point in clickData["points"]:
                    if 'customdata' in point:
                        (cve,version,severity,score) = point['customdata']
                        app.logger.info(f"CVE: {cve} Score = {score}")
                        # add vulnerabilities to the state
                        if cve not in patched_state:
                            patched_state[cve]=(version,severity,score,game_interval)

                app.logger.info(f"Total patch added {len(patched_state)}")

        return json.dumps(clickData, indent=2),patched_state

@app.callback(
    Output('data-attacks', 'data'),
    Output('card-body-score', 'children'),
    Input('game-interval', 'n_intervals'),
    Input('data-epss','data'),
    Input('data-attacks','data'),
    Input('data-patched', 'data'),
    )
def update_game_state(game_interval,epss_state,attack_state,defend_state):

    if attack_state is None: 
        attack_state = {}

    if game_interval > 0:
        is_attack = random.choice([True, False])
        if is_attack:
            app.logger.info(f'Creating attacks on day = {game_interval}')
            # calculate the probability of attack
            epss_df = pd.read_json(epss_state, orient='split')
            # do a draw for each day
            cves = random.choices(epss_df['cve'].to_list(),weights=epss_df.epss.to_list(),k=1)
            for cve in cves:
                if cve not in attack_state: 
                    attack_state[cve]=(game_interval)

            app.logger.info(f'Randomizing {len(epss_df)} vulns')
            app.logger.info(f'Attacks {len(attack_state)} vulns')

            #calculate your score
            attack_cve  = set(attack_state.keys())
            if defend_state is None:
                defend_cve = set()
            else:
                defend_cve = set(defend_state.keys())

            tp = len(attack_cve.intersection(defend_cve))
            fp = len(defend_cve - attack_cve)
            fn = len(attack_cve - defend_cve)
            precision = 1.0*tp/(tp+fp)
            recall = 1.0*tp/(tp+fn)
            cbodyp = []
            cbodyp.append(html.H4("Game Score", className="card-title"))
            cbodyp.append(
                html.P(
                    f'Precision = {precision}',
                    className="card-text"
                )
            )
            cbodyp.append(
                html.P(
                    f'True Positives = {tp}',
                    className="card-text"
                )
            )
            cbodyp.append(
                html.P(
                    f'False Positives = {fp}',
                    className="card-text"
                )
            )
            cbodyp.append(
                html.P(
                    f'False Negatives = {fn}',
                    className="card-text"
                )
            )
            app.logger.info(f'Precision {precision} ')
            return attack_state,cbodyp
        else:
            raise PreventUpdate

    else:
        raise PreventUpdate

@app.callback(
    Output('selected-data', 'children'),
    Input('basic-interactions', 'selectedData'))
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)


@app.callback(
    Output('hover-data', 'children'),
    Input('basic-interactions', 'hoverData'))
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)
    
@app.callback(
    Output('card-body-patched', 'children'),
    Input('data-patched', 'data'),
    Input('reset-button', 'n_clicks'))
def update_patch_data(patched_state,reset_button):

    ctx = dash.callback_context

    if patched_state is None: patched_state = {}
    cbodyp = []
    cbodyp.append(html.H4("Patched", className="card-title"))

    for trigger in ctx.triggered:
        if trigger['prop_id'] == 'reset-button.n_clicks':
            return cbodyp
        else:
            for cve, attr in patched_state.items():
                app.logger.info(f"K: {cve} V = {attr}")
                cbodyp.append(
                    html.P(
                        f"CVE = {cve} SCORE = {attr[2]} on day {attr[3]}",
                        className="card-text"
                    )
                )
    return cbodyp

@app.callback(
    Output('card-body-attacks', 'children'),
    Input('data-attacks', 'data'))
def update_attack_data(attack_state):
    if attack_state is None: attack_state = {}
    cbodya = []
    cbodya.append(html.H4("Attacks", className="card-title"))
    for cve, attr in attack_state.items():
        cbodya.append(
            html.P(
                f"CVE = {cve} on day {attr}",
                className="card-text"
            )
        )
    return cbodya


@app.callback(
    Output('relayout-data', 'children'),
    Input('basic-interactions', 'relayoutData'))
def display_relayout_data(relayoutData):
    return json.dumps(relayoutData, indent=2)


def add_cvss(cveid):
    r = nvdlib.getCVE(cveid)
    scores = r.score
    return scores

from os import path

def load_data(date='2022-03-01'):
    url = f'https://api.first.org/data/v1/epss?date={date}'

    if path.exists(f'./data/{date}.pkl'):
        app.logger.info('Loading dataset from disk cache')
        df_top = pd.read_pickle(f'./data/{date}.pkl')
    else:
        app.logger.info('Downloading EPSS day feed')
        r = requests.get(url)

        data = r.json()['data']

        df = pd.DataFrame(data)
        app.logger.info(f'Loaded {len(df)} CVE')
        app.logger.info('Correlating NVD scores...')
        df[['version','score','severity']]=df.apply(lambda x:add_cvss(x['cve']),axis=1, result_type="expand")
        app.logger.info('Done saving now...')
        df.to_pickle(f'./data/{date}.pkl')

    df_top['epss']=df_top['epss'].astype('float')
    df_top['percentile']=df_top['percentile'].astype('float')

    return df_top

@app.callback(
    Output("loading-spinner", "children"),
    Output('replay-label', 'children'),
    Output('basic-interactions', 'figure'),
    Output('data-epss','data'),
    Input('replay', 'value'))
def update_data(value):
    if value:
        epss_df = load_data()
        app.logger.info(f'Total vulns = {len(epss_df)}')

        fig = px.scatter(epss_df, x="epss", y="score", color="cve", custom_data=["cve","version","severity","score"])
        fig.update_layout(clickmode='event+select')
        fig.update_traces(marker_size=10)
        days_str = "Replay from {} day".format(value)

        return "EPSS data loaded",days_str,fig,epss_df.to_json(date_format='iso', orient='split')
    else:
        raise PreventUpdate

if __name__ == "__main__":
    app.run_server(debug=True,host= '0.0.0.0', port=8051)
