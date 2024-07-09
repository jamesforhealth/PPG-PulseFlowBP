import time
import sys
import logging
import os
from typing import Union
import dash
from dash.dependencies import Output, Input, State
from dash import html, dcc, ctx, dash_table
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import httpx
import numpy as np
from dataclasses import dataclass
from pydantic import BaseModel
import sqlite3
import h5py
import scipy
from pathlib import Path
# from matplotlib import pyplot as plt
# from mat73 import loadmat


# ENVIRONMENT VARIABLES; fixed for now
DEBUG_FLAG = False
SQLITE_FILE = 'PulseDB analysis test3.sqlite3'
PULSEDB_DIR = os.environ.get('PULSEDB_DIR')  # path to .mat files.  .mat files are located in this directory.  Change to switch between PulseDB - Vital and PulseDB - MIMIC
sample_rate = 125.0  # sample rate of the PulseDB .mat files

if PULSEDB_DIR:
    print("PULSEDB_DIR: {}".format(PULSEDB_DIR))
else:
    print("PULSEDB_DIR is not set.  Unable to view raw data.")


class Data(BaseModel):
    data_source: str = None
    array_index: int = None  # index from PulseDB .mat file
    sample_rate: Union[float, str] = None  # include str for '' (empty query result)

    patient_identifier: str = None  # patient identifier
    gender: str = None
    weight: float = None
    height: float = None
    age: int = None

    basic_analysis_id: int = None
    HR: float = None
    MAP: float = None
    PP: float = None
    SBP: float = None
    DBP: float = None

    normalized_wk3_analysis_id: int = None
    WK3n_R: float = None
    WK3n_C: float = None
    WK3n_Zc: float = None
    WK3n_RMSE: float = None
    WK3n_RMSE_systole: float = None
    WK3n_RMSE_diastole: float = None

    wk3_analysis_id: int = None
    WK3_R: float = None
    WK3_C: float = None
    WK3_Zc: float = None
    WK3_RMSE: float = None
    WK3_RMSE_systole: float = None
    WK3_RMSE_diastole: float = None


app = dash.Dash(__name__, update_title=None)

client = httpx.Client(timeout=60.0, follow_redirects=True)  # this takes some time to start up ~2s
app.layout = html.Div(
    [
        dcc.Interval(
            id="load_interval",
            n_intervals=0,
            max_intervals=0,  #<-- only run once
            interval=1
        ),
        html.H1('PulseDB analysis', style={'textAlign': 'center', 'color': '#000000'}),

        dcc.Loading(
            id="loading-datatable",
            type="circle",
            children=
            dash_table.DataTable(
                id='datatable',
                columns=[
                    {'id': 'data_source', 'name': 'data source'},
                    {'id': 'sample_rate', 'name': 'sample rate (Hz)'},

                    {'id': 'patient_identifier', 'name': 'patient'},
                    {'id': 'array_index', 'name': 'array index'},

                    {'id': 'gender', 'name': 'gender'},
                    {'id': 'weight', 'name': 'weight (kg)'},
                    {'id': 'height', 'name': 'height (cm)'},
                    {'id': 'age', 'name': 'age (years)'},

                    {'id': 'basic_analysis_id', 'name': 'basic analysis ID'},
                    {'id': 'HR', 'name': 'HR (bpm)'},
                    {'id': 'MAP', 'name': 'aortic MAP (mmHg)'},
                    {'id': 'PP', 'name': 'radial PP (mmHg)'},
                    {'id': 'SBP', 'name': 'radial SBP (mmHg)'},
                    {'id': 'DBP', 'name': 'radial DBP (mmHg)'},

                    {'id': 'normalized_wk3_analysis_id', 'name': 'normalized WK3 analysis ID'},
                    {'id': 'WK3n_R', 'name': 'WK3n_R'},
                    {'id': 'WK3n_C', 'name': 'WK3n_C'},
                    {'id': 'WK3n_Zc', 'name': 'WK3n_Zc'},
                    {'id': 'WK3n_RMSE', 'name': 'WK3n_RMSE'},
                    {'id': 'WK3n_RMSE_systole', 'name': 'WK3n_RMSE_systole'},
                    {'id': 'WK3n_RMSE_diastole', 'name': 'WK3n_RMSE_diastole'},

                    {'id': 'wk3_analysis_id', 'name': 'WK3 analysis ID'},
                    {'id': 'WK3_R', 'name': 'WK3_R'},
                    {'id': 'WK3_C', 'name': 'WK3_C'},
                    {'id': 'WK3_Zc', 'name': 'WK3_Zc'},
                    {'id': 'WK3_RMSE', 'name': 'WK3_RMSE'},
                    {'id': 'WK3_RMSE_systole', 'name': 'WK3_RMSE_systole'},
                    {'id': 'WK3_RMSE_diastole', 'name': 'WK3_RMSE_diastole'},
                ],

                page_current=0,
                page_size=20,
                page_action='native',
                sort_action='native',
                filter_action='native',
                cell_selectable=False,
                row_selectable="multi",
                style_table={
                    'overflowX': 'auto'
                },
                style_cell={
                    'textAlign': 'left'
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'patient'},
                     'width': 300},
                ],
                style_as_list_view=False,
                style_data={
                    'color': 'black',
                    'backgroundColor': 'white',
                    'whiteSpace': 'normal',
                    'height': 'auto'
                },
            ),
        ),  # END loading datatable
        html.Div(children=[html.Button(id='update-datatable-button', children='Update Data',
                                       style={'height': 50, 'fontSize': 28, 'textAlign': 'center', 'margin': 'auto'})],
                 style={'display': 'grid', 'margin': 'auto'}),
        html.Div(children=[html.Button(id='update-plot-button', children='Update Plot',
                                       style={'height': 50, 'fontSize': 28, 'textAlign': 'center', 'margin': 'auto'})],
                 style={'display': 'grid', 'margin': 'auto'}),
        html.Div(children=[
            dcc.Input(id='WK3n-RMSE-threshold', placeholder='WK3n_RMSE_threshold', type='number', value=0.1, step='any', min=0, style={'textAlign': 'center'}),
            dcc.RadioItems(id='WK3n-RMSE-threshold-radio', options=["plot greater than", "plot less than", "plot all selected"], value="plot all selected",)
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'padding': '10px'}),
        html.Div(children=[html.Button(id='update-radial-ABP-segment-button', children='Update Radial ABP Raw Segment',
                                       style={'height': 50, 'fontSize': 28, 'textAlign': 'center', 'margin': 'auto'})],
                 style={'display': 'grid', 'margin': 'auto'}),
        dcc.Loading(
            id="loading-plot",
            type="circle",
            children=dcc.Graph(id='plot', animate=False),
        ),
        dcc.Loading(
            id="loading-plot2",
            type="circle",
            children=dcc.Graph(id='plot-WK3n-RMSE', animate=False),
        ),
        dcc.Loading(
            id="loading-plot3",
            type="circle",
            children=dcc.Graph(id='plot-radial-ABP-raw-segment', animate=False),
        )
    ],
    style={'backgroundColor': '#BF40BF'},

)


# @app.callback(Output('datatable', 'data'),
#               Input(component_id="load_interval", component_property="n_intervals")
#               )
# def init_datatable(x):
#     item1 = Data(
#         patient='patient1',
#         gender="male",
#         weight=70,
#         height=170,
#         age=30,
#         data_segment_id=1,
#         sample_rate=100,
#         basic_analysis_id=1,
#         HR=70,
#         MAP=70,
#         PP=70,
#         SBP=70,
#         DBP=70,
#         normalized_wk3_analysis_id=1,
#         WK3n_R=70,
#         WK3n_C=70,
#         WK3n_Zc=70,
#         wk3_analysis_id=1,
#         WK3_R=70,
#         WK3_C=70,
#         WK3_Zc=70
#     )
#     item2 = Data(
#         patient='patient2',
#         gender="female",
#         weight=80,
#         height=150,
#         age=40,
#         data_segment_id=2,
#         sample_rate=200,
#         basic_analysis_id=2,
#         HR=90,
#         MAP=90,
#         PP=90,
#         SBP=90,
#         DBP=80,
#         normalized_wk3_analysis_id=2,
#         WK3n_R=70,
#         WK3n_C=70,
#         WK3n_Zc=70,
#         wk3_analysis_id=2,
#         WK3_R=70,
#         WK3_C=70,
#         WK3_Zc=70
#     )
#     out = [item1.dict(), item2.dict()]
#     # print(out)
#     return out

def query(query: str, database_file: str = 'PulseDB analysis test.sqlite3'):
    cnx = sqlite3.connect(SQLITE_FILE)
    cursor = cnx.cursor()
    cursor.execute(query)  # print all tables in the database
    results = cursor.fetchall()
    cursor.close()
    cnx.close()

    return results


@app.callback(
    Output('datatable', 'data'),
    [
        Input('update-datatable-button', 'n_clicks')
    ],
    [
        State('datatable', 'data')
    ],
    prevent_initial_call=False
)
def update_datatable(n_clicks, data):
    print('>> update datatable <<')

    out = {'data': None, 'layout': go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')}

    # query1 = "SELECT name FROM sqlite_master WHERE type='table';"
    # result1 = query(query1)
    # print(result1)

    # query2 = "SELECT * FROM patient_info;"
    # result2 = query(query2)
    # for patient in result2:
    #     print(patient)

    query3 = """
    SELECT
    ds.name,
    d.array_index,
    d.sample_rate_hz,
    
    p.identifier,
    g.gender,
    p.age,
    p.height_cm,
    p.weight_kg,
    
    ba.heart_rate_bpm,
    ba.systolic_pressure_mmhg,
    ba.diastolic_pressure_mmhg,
    ba.mean_arterial_pressure_mmhg,
    ba.pulse_pressure_mmhg,
    
    nwa.R,
    nwa.C,
    nwa.Zc,
    nwa.root_mean_squared_error,
    nwa.root_mean_squared_error_systole,
    nwa.root_mean_squared_error_diastole,
    
    wa.R,
    wa.C,
    wa.Zc,
    wa.root_mean_squared_error,
    wa.root_mean_squared_error_systole,
    wa.root_mean_squared_error_diastole
    FROM data_segment d
    INNER JOIN patient_info_snapshot p ON d.patient_snapshot_id = p.id
    INNER JOIN gender g on p.gender = g.id
    INNER JOIN data_source ds on d.data_source = ds.id
    LEFT JOIN basic_analysis ba ON d.id = ba.segment_id
    LEFT JOIN normalized_wk3_analysis nwa ON d.id = nwa.segment_id
    LEFT JOIN wk3_analysis wa ON d.id = wa.segment_id

    
    """
    # LEFT JOIN patient pp ON pp.id = p.patient_id

    result3 = query(query3)
    if DEBUG_FLAG:
        print(query3)
    # print(result3)
    table_results = []

    for result in result3:
        # replace None with NaN for analysis columns
        result = list(result)
        result[8::] = [np.nan if x is None else x for x in result[8::]]

        if DEBUG_FLAG:
            print(result)
        item = Data(
            data_source=result[0],
            array_index=result[1],
            sample_rate=result[2],

            patient_identifier=result[3],
            gender=result[4],
            age=result[5],
            height=result[6],
            weight=result[7],

            HR=result[8],
            SBP=result[9],
            DBP=result[10],
            MAP=result[11],
            PP=result[12],

            WK3n_R=result[13],
            WK3n_C=result[14],
            WK3n_Zc=result[15],
            WK3n_RMSE=result[16],
            WK3n_RMSE_systole=result[17],
            WK3n_RMSE_diastole=result[18],

            WK3_R=result[19],
            WK3_C=result[20],
            WK3_Zc=result[21],
            WK3_RMSE=result[22],
            WK3_RMSE_systole=result[23],
            WK3_RMSE_diastole=result[24]
        )
        table_results.append(item.dict())

    return table_results


def interpolate(x: list, y: list, h=1.0) -> tuple[list, list]:
    # interpolate from min(x) to max(x) using step size h
    # f = scipy.interpolate.PchipInterpolator(np.array(x), np.array(y))
    x_new = np.arange(min(x), max(x), h)
    # y_new = f(x_new)
    return x_new.tolist(), []  # y_new.tolist()


@app.callback(
    Output('plot', 'figure'),
    [
        Input('update-plot-button', 'n_clicks'),
        Input('datatable', 'data')  # callback chain: update_datatable -> update_plot
    ],
    State('datatable', 'data'),
    prevent_initial_call=True
)
def update_plot(n_clicks, _, table_results):
    print('>> update plot <<')
    if DEBUG_FLAG:
        print(table_results)

    # plot WK3n
    out = make_subplots(rows=4, cols=3, shared_xaxes=False, shared_yaxes=False,
                        subplot_titles=[])
    out.update_layout(
        title=dict(
            text="normalized WK3 analysis | text=RMSE",
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        height=400 * 6
    )

    out.add_trace(go.Scatter(x=[item['MAP'] for item in table_results], y=[item['WK3n_R'] for item in table_results],
                             name='WK3n_R vs MAP',
                             mode='markers', marker=dict(symbol='circle-open', color='white', opacity=0.7, size=10,
                                                         line=dict(width=2))), row=1, col=1)
    out.update_xaxes(
        title_text='MAP (mmHg)',
        row=1, col=1,
        showgrid=True,
        tickvals=np.arange(0, 1000, 20),
        range=[0, 200]
    )
    out.update_yaxes(
        title_text='R',
        row=1, col=1,
        showgrid=True,
        tickvals=np.arange(0, 20, 1),
        range=[0, 5]
    )

    out.add_trace(go.Scatter(x=[item['PP'] for item in table_results], y=[item['WK3n_R'] for item in table_results],
                             name='WK3n_R vs PP',
                             mode='markers', marker=dict(symbol='circle-open', color='white', opacity=0.7, size=10,
                                                         line=dict(width=2))), row=2, col=1)
    out.update_xaxes(
        title_text='PP (mmHg)',
        row=2, col=1,
        showgrid=True,
        tickvals=np.arange(0, 1000, 10),
        range=[0, 100]
    )
    out.update_yaxes(
        title_text='R',
        row=2, col=1,
        showgrid=True,
        tickvals=np.arange(0, 20, 1),
        range=[0, 5]
    )

    out.add_trace(go.Scatter(x=[item['MAP'] for item in table_results], y=[item['WK3n_C'] for item in table_results],
                             text=["{:.2f}".format(item['WK3n_RMSE']) if item['WK3n_RMSE'] else "" for item in table_results],
                             name='WK3n_C vs MAP', textposition="bottom center",
                             mode='markers', marker=dict(symbol='circle-open', color='white', opacity=0.7, size=10,
                                                         line=dict(width=2))), row=1, col=2)
    out.update_xaxes(
        title_text='MAP (mmHg)',
        row=1, col=2,
        showgrid=True,
        tickvals=np.arange(0, 1000, 20),
        range=[0, 200]
    )
    out.update_yaxes(
        title_text='C',
        row=1, col=2,
        showgrid=True,
        # tickvals=np.arange(0, 1000, 50),
        # range=[0, 300]
    )

    out.add_trace(go.Scatter(x=[item['PP'] for item in table_results], y=[item['WK3n_C'] for item in table_results],
                             name='WK3n_C vs PP',
                             mode='markers', marker=dict(symbol='circle-open', color='white', opacity=0.7, size=10,
                                                         line=dict(width=2))), row=2, col=2)
    out.update_xaxes(
        title_text='PP (mmHg)',
        row=2, col=2,
        showgrid=True,
        tickvals=np.arange(0, 1000, 10),
        range=[0, 100]
    )
    out.update_yaxes(
        title_text='C',
        row=2, col=2,
        showgrid=True,
        # tickvals=np.arange(0, 1000, 50),
        # range=[0, 300]
    )

    out.add_trace(go.Scatter(x=[item['MAP'] for item in table_results], y=[item['WK3n_Zc'] for item in table_results],
                             name='WK3n_Zc vs MAP',
                             mode='markers', marker=dict(symbol='circle-open', color='white', opacity=0.7, size=10,
                                                         line=dict(width=2))), row=1, col=3)
    out.update_xaxes(
        title_text='MAP (mmHg)',
        row=1, col=3,
        showgrid=True,
        tickvals=np.arange(0, 1000, 20),
        range=[0, 200]
    )
    out.update_yaxes(
        title_text='Zc',
        row=1, col=3,
        showgrid=True,
        # tickvals=np.arange(0, 1000, 50),
        # range=[0, 300]
    )

    out.add_trace(go.Scatter(x=[item['PP'] for item in table_results], y=[item['WK3n_Zc'] for item in table_results],
                             name='WK3n_Zc vs PP',
                             mode='markers', marker=dict(symbol='circle-open', color='white', opacity=0.7, size=10,
                                                         line=dict(width=2))), row=2, col=3)
    out.update_xaxes(
        title_text='PP (mmHg)',
        row=2, col=3,
        showgrid=True,
        tickvals=np.arange(0, 1000, 10),
        range=[0, 100]
    )
    out.update_yaxes(
        title_text='Zc',
        row=2, col=3,
        showgrid=True,
        # tickvals=np.arange(0, 1000, 50),
        # range=[0, 300]
    )

    out.add_trace(go.Scatter(x=[item['HR'] for item in table_results], y=[item['PP'] for item in table_results],
                             name='PP vs HR',
                             mode='markers', marker=dict(symbol='circle-open', color='white', opacity=0.7, size=10,
                                                         line=dict(width=2))), row=3, col=1)
    out.update_xaxes(
        title_text='HR (bpm)',
        row=3, col=1,
        showgrid=True,
        tickvals=np.arange(0, 200, 20),
        range=[0, 200]
    )
    out.update_yaxes(
        title_text='PP (mmHg)',
        row=3, col=1,
        showgrid=True,
        # tickvals=np.arange(0, 1000, 50),
        # range=[0, 300]
    )

    out.add_trace(go.Scatter(x=[item['age'] for item in table_results], y=[item['MAP'] for item in table_results],
                             name='MAP vs age',
                             mode='markers', marker=dict(symbol='circle-open', color='white', opacity=0.7, size=10,
                                                         line=dict(width=2))), row=3, col=2)
    out.update_xaxes(
        title_text='age (years)',
        row=3, col=2,
        showgrid=True,
        tickvals=np.arange(0, 100, 20),
        range=[0, 100]
    )
    out.update_yaxes(
        title_text='MAP (mmHg)',
        row=3, col=2,
        showgrid=True,
        # tickvals=np.arange(0, 1000, 50),
        # range=[0, 300]
    )

    out.add_trace(go.Scatter(x=[item['age'] for item in table_results], y=[item['WK3n_C'] for item in table_results],
                             name='WK3n_C vs age',
                             mode='markers', marker=dict(symbol='circle-open', color='white', opacity=0.7, size=10,
                                                         line=dict(width=2))), row=3, col=3)
    out.update_xaxes(
        title_text='age (years)',
        row=3, col=3,
        showgrid=True,
        tickvals=np.arange(0, 100, 20),
        range=[0, 100]
    )
    out.update_yaxes(
        title_text='C',
        row=3, col=3,
        showgrid=True,
        # tickvals=np.arange(0, 1000, 50),
        # range=[0, 300]
    )

    MAP = [item['MAP'] for item in table_results]
    PP = [item['PP'] for item in table_results]
    WK3n_C = [item['WK3n_C'] for item in table_results]
    WK3n_Zc = [item['WK3n_Zc'] for item in table_results]
    WK3n_RMSE = [item['WK3n_RMSE'] for item in table_results]
    data_source = [item['data_source'] for item in table_results]
    patient_identifier = [item['patient_identifier'] for item in table_results]
    zipped = list(zip(data_source, patient_identifier, MAP, PP, WK3n_C, WK3n_Zc, WK3n_RMSE))
    zipped_sorted_unique_patient = sorted(zipped, key=lambda x: str(x[0]) + " " + str(x[1]))  # ex. PulseDB - Vital p001234

    patient_set = set()
    for item in zipped_sorted_unique_patient:
        if DEBUG_FLAG:
            print(item)
        # add a unique patient across data sources to the set.  ex. (PulseDB - Vital, p001234)
        patient_set.add(item[0] + " @ " + item[1])
    for ds_patient in sorted(patient_set):
        if DEBUG_FLAG:
            (ds_patient)
        # scan zipped for this unique patient's data and append to list
        curr_MAP = []
        curr_PP = []
        curr_WK3n_C = []
        curr_WK3n_Zc = []
        curr_RMSE = []
        for item in zipped_sorted_unique_patient:
            if item[0] + " @ " + item[1] == ds_patient:
                # if WK3n_C is not None, append to list
                if item[4] is not None:
                    curr_MAP.append(item[2])
                    curr_PP.append(item[3])
                    curr_WK3n_C.append(item[4])
                    curr_WK3n_Zc.append(item[5])
                    curr_RMSE.append(item[6])
        if DEBUG_FLAG:
            (list(zip(curr_MAP, curr_WK3n_C)))
        sorted_MAP, sorted_WK3n_C_MAP = zip(*sorted(list(zip(curr_MAP, curr_WK3n_C)), key=lambda x: x[0]))
        sorted_PP, sorted_WK3n_C_PP = zip(*sorted(list(zip(curr_PP, curr_WK3n_C)), key=lambda x: x[0]))
        MAP_new, dWK3n_C_dMAP = interpolate(sorted_MAP, sorted_WK3n_C_MAP)
        PP_new, dWK3n_C_dPP = interpolate(sorted_PP, sorted_WK3n_C_PP)
        # dWK3n_C_dMAP = np.gradient(dWK3n_C_dMAP)
        # dWK3n_C_dPP = np.gradient(dWK3n_C_dPP)

        # do linear regression if more than one data point
        if len(curr_MAP) > 1:
            # do linear regression for MAP -> C
            MAP_C_regression_result = scipy.stats.linregress(curr_MAP, curr_WK3n_C)
            if DEBUG_FLAG:
                print(MAP_C_regression_result)
            MAP_C_linear_fit = MAP_C_regression_result.intercept + MAP_C_regression_result.slope * np.array(MAP_new)  # borrow MAP_new to do interpolation

            # do linear regression for PP -> C
            PP_C_regression_result = scipy.stats.linregress(curr_PP, curr_WK3n_C)
            if DEBUG_FLAG:
                print(PP_C_regression_result)
            PP_linear_fit = PP_C_regression_result.intercept + PP_C_regression_result.slope * np.array(PP_new)  # borrow PP_new to do interpolation
            
            # do linear regression for MAP -> Zc
            MAP_Zc_regression_result = scipy.stats.linregress(curr_MAP, curr_WK3n_Zc)
            if DEBUG_FLAG:
                print(MAP_Zc_regression_result)
            MAP_Zc_linear_fit = MAP_Zc_regression_result.intercept + MAP_Zc_regression_result.slope * np.array(MAP_new)

            # out.add_trace(go.Scatter(x=MAP_new, y=dWK3n_C_dMAP,
            #                          name='d WK3n_C / d MAP' + " @ " + ds_patient + " // r value: " + "{:.2f}".format(MAP_C_regression_result.rvalue),
            #                          mode='lines', marker=dict(symbol='circle-open', color='yellow', opacity=0.7, size=10,
            #                                                    line=dict(width=1))), row=1, col=1)

            opacity0 = np.min([np.abs(MAP_C_regression_result.rvalue), np.abs(PP_C_regression_result.rvalue)])
            if opacity0 < 0.65:
                opacity = 0.2
            else:
                opacity = opacity0
            # print(opacity)
            out.add_trace(go.Scatter(x=MAP_new, y=MAP_C_linear_fit,
                                     name='WK3n_C vs MAP' + " @ " + ds_patient + " // r value: " + "{:.2f}".format(MAP_C_regression_result.rvalue),
                                     mode='lines', marker=dict(symbol='circle-open', color='yellow', opacity=opacity, size=10),
                                     line=dict(width=1, color='rgba(255, 255, 0, {})'.format(opacity))), row=1, col=2)
            out.add_trace(go.Scatter(x=curr_MAP, y=curr_WK3n_C,
                                     text=["{:.2f}".format(item) if item else "" for item in curr_RMSE],
                                     name='WK3n_C vs MAP' + " @ " + ds_patient, textposition="bottom center",
                                     mode='markers+text',
                                     marker=dict(symbol='x', color='yellow', opacity=opacity, size=5),
                                     textfont=dict(color='rgba(0, 0, 0, {})'.format(opacity))), row=1, col=2)
            out.add_trace(go.Scatter(x=MAP_new, y=MAP_Zc_linear_fit,
                                     name='WK3n_Zc vs MAP' + " @ " + ds_patient + " // r value: " + "{:.2f}".format(MAP_Zc_regression_result.rvalue),
                                     mode='lines',
                                     marker=dict(symbol='circle-open', color='yellow', opacity=opacity, size=10),
                                     line=dict(width=1, color='rgba(0, 255, 0, {})'.format(opacity))), row=1, col=3)
            out.add_trace(go.Scatter(x=curr_MAP, y=curr_WK3n_Zc,
                                     text=["{:.2f}".format(item) if item else "" for item in curr_RMSE],
                                     name='WK3n_Zc vs MAP' + " @ " + ds_patient, textposition="bottom center",
                                     mode='markers+text',
                                     marker=dict(symbol='x', color='green', opacity=opacity, size=5),
                                     textfont=dict(color='rgba(0, 0, 0, {})'.format(opacity))), row=1, col=3)

            out.add_trace(go.Scatter(x=PP_new, y=PP_linear_fit,
                                     name='WK3n_C vs PP' + " @ " + ds_patient + " // r value: " + "{:.2f}".format(PP_C_regression_result.rvalue),
                                     mode='lines', marker=dict(symbol='circle-open', color='yellow', opacity=opacity, size=10),
                                     line=dict(width=1, color='rgba(255, 255, 0, {})'.format(opacity))), row=2, col=2)
            out.add_trace(go.Scatter(x=curr_PP, y=curr_WK3n_C,
                                     text=["{:.2f}".format(item) if item else "" for item in curr_RMSE],
                                     name='WK3n_C vs PP' + " @ " + ds_patient, textposition="bottom center",
                                     mode='markers+text', marker=dict(symbol='x', color='yellow', opacity=opacity, size=5),
                                     textfont=dict(color='rgba(0, 0, 0, {})'.format(opacity))), row=2, col=2)



            # zipped_MAP_WK3n_C = list(zip(MAP, WK3n_C))
            # sorted_zipped_MAP_WK3n_C = sorted(zipped_MAP_WK3n_C, key=lambda x: x[0])
            # MAP_sorted, WK3n_C_sorted = zip(*sorted_zipped_MAP_WK3n_C)
            # f_WK3n_C_interp = scipy.interpolate.PchipInterpolator(MAP_sorted, WK3n_C_sorted)
            # MAP_new = np.arange(np.min(MAP), np.max(MAP), 1.0)
            # dWK3n_C_dMAP = np.gradient(f_WK3n_C_interp(MAP_new))
            # # TODO: the above sorts using all patients' data; want: for each patient, sort using only that patient's data


            # plot derivative // not good idea, due to noise
            # out.add_trace(go.Scatter(x=MAP_new, y=dWK3n_C_dMAP,
            #                          name='dWK3n_C/dMAP' + " @ " + ds_patient,
            #                          mode='lines', marker=dict(symbol='circle-open', color='white', opacity=0.7, size=10,
            #                                                    line=dict(width=1))), row=4, col=2)
            # out.update_xaxes(
            #     title_text='MAP (mmHg)',
            #     row=4, col=2,
            #     showgrid=True,
            #     tickvals=np.arange(0, 1000, 20),
            #     range=[0, 200]
            # )
            # out.update_yaxes(
            #     title_text='dC/dMAP',
            #     row=4, col=2,
            #     showgrid=True,
            #     # tickvals=np.arange(0, 1000, 50),
            #     # range=[0, 300]
            # )

    # ------------
    return out


@app.callback(
    Output('plot-WK3n-RMSE', 'figure'),
    [
        Input('update-plot-button', 'n_clicks'),
        Input('datatable', 'data')  # callback chain: update_datatable -> update_plot
    ],
    State('datatable', 'data'),
    prevent_initial_call=True
)
def update_WK3n_RMSE_plot(n_clicks, _, table_results):
    WK3n_RMSE = [item['WK3n_RMSE'] for item in table_results]
    WK3n_RMSE_systole = [item['WK3n_RMSE_systole'] for item in table_results]
    WK3n_RMSE_diastole = [item['WK3n_RMSE_diastole'] for item in table_results]

    out = make_subplots(rows=1, cols=1, shared_xaxes=False, shared_yaxes=False,
                        subplot_titles=[])
    out.update_layout(
        title=dict(
            text="normalized WK3 RMSE",
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        height=400 * 1,
        barmode='stack'
    )
    # out.add_trace(go.Scatter(x=[item['MAP'] for item in table_results], y=[item['WK3n_R'] for item in table_results],
    #                          name='RMSE histogram'), row=1, col=1)
    out.add_trace(go.Histogram(x=WK3n_RMSE, name='WK3n_RMSE', nbinsx=100), row=1, col=1)
    out.add_trace(go.Histogram(x=WK3n_RMSE_systole, name='WK3n_RMSE_systole', nbinsx=100), row=1, col=1)
    out.add_trace(go.Histogram(x=WK3n_RMSE_diastole, name='WK3n_RMSE_diastole', nbinsx=100), row=1, col=1)
    out.update_xaxes(
        title_text='RMSE',
        row=1, col=1,
        showgrid=True,
        # tickvals=np.arange(0, 1000, 20),
        # range=[0, 200]
    )
    out.update_yaxes(
        title_text='count',
        row=1, col=1,
        showgrid=True,
        # tickvals=np.arange(0, 20, 1),
        # range=[0, 5]
    )

    return out


@app.callback(
    Output('plot-radial-ABP-raw-segment', 'figure'),
    Input('update-radial-ABP-segment-button', 'n_clicks'),
    [
        State('datatable', 'data'),
        State('datatable', 'selected_rows'),
        State('WK3n-RMSE-threshold', 'value'),
        State('WK3n-RMSE-threshold-radio', 'value')
    ]
)
def update_eye_diagram_radial_ABP_plot(n_clicks, table_results, selected_rows: list, WK3n_RMSE_threshold, WK3n_RMSE_threshold_radio):
    out = make_subplots(rows=1, cols=1, shared_xaxes=False, shared_yaxes=False,
                        subplot_titles=[])
    if WK3n_RMSE_threshold_radio == "plot all selected":
        title = "Radial ABP Raw Segment: <data source>/<patient identifier>/<array index> | <WK3n RMSE>[selected]"
    elif WK3n_RMSE_threshold_radio == "plot greater than":
        title = "Radial ABP Raw Segment: <data source>/<patient identifier>/<array index> | <WK3n RMSE>[>{:.4f}]".format(WK3n_RMSE_threshold)
    elif WK3n_RMSE_threshold_radio == "plot less than":
        title = "Radial ABP Raw Segment: <data source>/<patient identifier>/<array index> | <WK3n RMSE>[<{:.4f}]".format(WK3n_RMSE_threshold)
    else:
        title = ""
    out.update_layout(
        title=dict(
            text=title, # "Radial ABP Raw Segment: <data source>/<patient identifier>/<array index> | <WK3n RMSE>",
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,.9)',
        showlegend=True,
        height=700 * 1
    )
    color = 'rgba(255, 0, 0, 0.2)'
    out.update_xaxes(showgrid=True, zeroline=False, gridcolor=color, title_text='time (s)',
                     range=[-1.5, 1.5], tickvals=np.arange(-100, 100, dtype=np.int64)/10)
    out.update_yaxes(showgrid=True, zeroline=False, gridcolor=color, title_text='ABP (mmHg)',
                     range=[0, 200], tickvals=np.arange(0, 300, 20, dtype=np.int64))

    # for Scatter3d
    # out.update_layout(scene=dict(
    #     xaxis=dict(title='', showgrid=True, zeroline=False, gridcolor=color),
    #     yaxis=dict(title='time (s)', showgrid=True, zeroline=False, gridcolor=color),
    #     zaxis=dict(title='ABP (mmHg)', showgrid=True, zeroline=False, gridcolor=color)
    # ))
    # out.layout.scene.camera.projection.type = 'orthographic'

    if DEBUG_FLAG:
        print(WK3n_RMSE_threshold)
    if WK3n_RMSE_threshold_radio == "plot all selected":
        if selected_rows is None:
            selected_rows = []
        else:
            selected_rows.sort()
    elif WK3n_RMSE_threshold_radio == "plot greater than":
        # get all rows with RMSE greater than threshold
        WK3n_RMSE = [item['WK3n_RMSE'] for item in table_results]
        selected_rows = [i for i, x in enumerate(WK3n_RMSE) if WK3n_RMSE[i] and WK3n_RMSE[i] > WK3n_RMSE_threshold]
    elif WK3n_RMSE_threshold_radio == "plot less than":
        # get all rows with RMSE less than threshold
        WK3n_RMSE = [item['WK3n_RMSE'] for item in table_results]
        selected_rows = [i for i, x in enumerate(WK3n_RMSE) if WK3n_RMSE[i] and WK3n_RMSE[i] < WK3n_RMSE_threshold]
        pass
    else:
        selected_rows = []
    if DEBUG_FLAG:
        print(selected_rows)

    count = 0
    for i in selected_rows:
        # print(table_results[i])
        data_source = table_results[i]['data_source']
        patient_identifier = table_results[i]['patient_identifier']  # "p001234"
        array_index = table_results[i]['array_index'] - 1  # convert to 0-index
        WK3n_RMSE_value = table_results[i]['WK3n_RMSE']
        if DEBUG_FLAG:
            print("{}/{}/{} | {}".format(data_source, patient_identifier, array_index, WK3n_RMSE_value))

        file = patient_identifier + ".mat"
        filename = str(Path(PULSEDB_DIR) / file)

        # tic = time.perf_counter()  # BEGIN TIMER
        # matdata = loadmat(filename)
        # toc1 = time.perf_counter()
        # matdata = matdata['Subj_Wins']
        # # SegmentID = matdata['SegmentID']
        # # SegmentID = np.array(SegmentID).flatten()
        # ABP_Raw = matdata['ABP_Raw'][array_index]  # list of lists
        # # ABP_Raw = np.array(ABP_Raw).flatten()
        # ABP_SPeaks = matdata['ABP_SPeaks']  # list of lists
        # ABP_SPeaks = np.array(ABP_SPeaks[array_index]).flatten().astype(np.int64) - 1 - 2  # convert to 0-index and remove 4th order filter latency
        # middle_peak = ABP_SPeaks[round(len(ABP_SPeaks) / 2)]
        # t_offset = middle_peak/sample_rate
        # toc = time.perf_counter()
        # print(toc-tic)  # END TIMER // mat73 is super slow; don't use


        ABP_Raw = []
        ABP_SPeaks = []
        # tic = time.perf_counter()  # BEGIN TIMER
        with h5py.File(filename, 'r') as f:
            matdata = f['Subj_Wins']
            items = zip(matdata['ABP_Raw'][0], matdata['ABP_SPeaks'][0])
            for _ABP_Raw, _ABP_SPeaks in items:
                ABP_Raw.append(f[_ABP_Raw][0])
                ABP_SPeaks.append(f[_ABP_SPeaks][0])
        # ABP_Raw = np.array(ABP_Raw).flatten()
        ABP_SPeaks = np.array(ABP_SPeaks[array_index]).flatten().astype(np.int64) - 1 - 2  # convert to 0-index and remove 4th order filter latency
        middle_peak = ABP_SPeaks[round(len(ABP_SPeaks) / 2)]
        t_offset = middle_peak / sample_rate
        # toc = time.perf_counter()
        # print("get segment from .mat time (s): {}".format(toc - tic))  # END TIMER

        # print(SegmentID)
        # print(ABP_Raw)
        # print(ABP_SPeaks)

        # get segment and plot
        if WK3n_RMSE_value:
            ABP_Raw_segment = np.array(ABP_Raw[array_index]).flatten()
            t = np.arange(len(ABP_Raw_segment))/sample_rate - t_offset
            out.add_trace(go.Scatter(x=t, y=ABP_Raw_segment, mode='lines', name="{}/{}/{} | {:.4f}".format(data_source, patient_identifier, array_index, WK3n_RMSE_value)))
            # out.add_trace(go.Scatter3d(x=np.zeros_like(t) + count, y=t, z=ABP_Raw_segment, mode='lines', name="{}/{}/{}".format(data_source, patient_identifier, array_index)))
            # out.add_trace(go.Scatter(x=t[ABP_SPeaks], y=ABP_Raw_segment[ABP_SPeaks], mode='markers', name="{}/{}/{}".format(data_source, patient_identifier, array_index)))
            # count += 1





    return out

if __name__ == '__main__':
    port = 8000
    app.run_server(debug=True, host="localhost", port=port)


    # import plotly.express as px
    # filename = str(Path(PULSEDB_DIR) / "p000003.mat")
    # ABP_Raw = []
    # SegSBP = []
    # SegDBP = []
    # ABP_SPeaks = []
    # with h5py.File(filename, 'r') as f:
    #     print(f.keys())
    #     print(f['Subj_Wins'].keys())  # <KeysViewHDF5 ['ABP_F', 'ABP_Lag', 'ABP_Raw', 'ABP_SPeaks', 'ABP_Turns', 'Age', 'BMI', 'CaseID', 'ECG_F', 'ECG_RPeaks', 'ECG_Raw', 'Gender', 'Height', 'IncludeFlag', 'PPG_ABP_Corr', 'PPG_F', 'PPG_Raw', 'PPG_SPeaks', 'PPG_Turns', 'SegDBP', 'SegSBP', 'SegmentID', 'SubjectID', 'Weight', 'WinID', 'WinSeqID']>
    #     data = f['Subj_Wins']
    #     print(data)
    #
    #     # age_dataset = data['Age']
    #     # print(age_dataset)
    #     # for item in age_dataset[0]:
    #     #     print(item)
    #     #     age_value = f[item][0]  # get age value
    #     #
    #     # ABP_dataset = data['ABP_Raw']
    #     # for item in ABP_dataset[0]:
    #     #     print(item)
    #     #     ABP_value = f[item][0]  # get ABP value (numpy.ndarray)
    #     #     print(ABP_value)
    #     #
    #     # SegmentID_dataset = data['SegmentID']
    #     # for item in SegmentID_dataset[0]:
    #     #     # print(item)
    #     #     SegmentID_value = f[item][0]  # get SegmentID value (numpy.ndarray)
    #     #     # print(SegmentID_value)
    #
    #     items = zip(data['CaseID'][0], data['SegmentID'][0], data['WinSeqID'][0], data['WinID'][0], data['ABP_Raw'][0], data['ABP_SPeaks'][0])
    #     for _CaseID, _SegmentID, _WinSeqID, _WinID, _ABP_Raw, _ABP_SPeaks in items:
    #         print(str(f[_CaseID][0]), str(f[_SegmentID][0]), str(f[_WinSeqID][0]), str(f[_WinID][0]), str(f[_ABP_Raw][0]))
    #         ABP_Raw.append(f[_ABP_Raw][0])
    #         ABP_SPeaks.append(f[_ABP_SPeaks][0])
    #
    #     items = zip(data['SegSBP'][0], data['SegDBP'][0])
    #     for _SegSBP, _SegDBP in items:
    #         # print(str(f[_SegSBP][0]), str(f[_SegDBP][0]))
    #         SegSBP.append(f[_SegSBP][0])
    #         SegDBP.append(f[_SegDBP][0])
    #
    # ABP_Raw = np.array(ABP_Raw).flatten()
    # SegSBP = np.array(SegSBP).flatten()
    # SegDBP = np.array(SegDBP).flatten()
    #
    # def get_zero_indexed_peaks(x):
    #     out = []
    #
    #     for i in range(len(x)):
    #         row = x[i]
    #         row_offset = [int(row_item + 1250*i) - 1 - 2 for row_item in row]  # convert to zero-indexed; also remove 4th order filter delay
    #         for j in range(len(row)):
    #             out.append(row_offset[j])
    #
    #     return out
    #
    # ABP_SPeaks = get_zero_indexed_peaks(ABP_SPeaks)
    # # print(ABP_SPeaks)
    #
    # # matplotlib is slow
    # # plt.figure()
    # # plt.plot(ABP_Raw)
    # # plt.show()
    #
    # t = np.arange(len(ABP_Raw))/125.0
    # dABP = np.gradient(ABP_Raw)*10
    # ddABP = np.gradient(np.gradient(ABP_Raw))*100
    # fig = go.Figure()
    # fig.update_xaxes(title_text='time (s)')
    # fig.update_yaxes(title_text='ABP (mmHg)')
    # fig.add_trace(go.Scatter(x=t, y=ABP_Raw, name='ABP', mode='lines'))
    # fig.add_trace(go.Scatter(x=t[ABP_SPeaks], y=ABP_Raw[ABP_SPeaks], mode='markers', name='ABP peaks', marker=dict(symbol='x', color='black', size=5)))
    # fig.add_trace(go.Scatter(x=t, y=dABP, name='dABP', mode='lines'))
    # fig.add_trace(go.Scatter(x=t[ABP_SPeaks], y=dABP[ABP_SPeaks], mode='markers', name='dABP peaks', marker=dict(symbol='x', color='black', size=5)))
    # fig.add_trace(go.Scatter(x=t, y=ddABP, mode='lines', name='ddABP'))
    # fig.add_trace(go.Scatter(x=t[ABP_SPeaks], y=ddABP[ABP_SPeaks], mode='markers', name='ddABP peaks', marker=dict(symbol='x', color='black', size=5)))
    # fig.show()
    #
    # fig2 = go.Figure()
    # fig2.add_trace(go.Scatter(x=np.arange(len(SegSBP)), y=SegSBP, mode='lines', name='SBP'))
    # fig2.add_trace(go.Scatter(x=np.arange(len(SegDBP)), y=SegDBP, mode='lines', name='DBP'))
    # fig2.show()

    # # try using mat73 // much easier than h5py
    # from mat73 import loadmat
    # data = loadmat(filename)
    # data = data['Subj_Wins']
    # ABP_Raw = data['ABP_Raw']  # list of lists
    # print(ABP_Raw)
    # ABP_SPeaks = data['ABP_SPeaks']  # list of lists
    # print(ABP_SPeaks)




# TODO: automate analysis in MATLAB app.  Then, re-analyze all segments and add point detection results to the database: {root, shoulder, notch}
# TODO MATLAB: add ddABP minima after notch --> flow = 0.  (Notch -> flow reaches minima)
