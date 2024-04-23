from pywebio import start_server
from pywebio.input import input, FLOAT
from pywebio.output import put_text
import pandas as pd
import numpy as np
import pulp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump, load

def predict_net_flow():
    
    data_df = pd.read_csv('Merged_Data_202101_202303_v3.csv', index_col=0)
    next_24_hours = pd.date_range(start=pd.Timestamp('now').floor('H'), periods=24, freq='H')
    all_stations = data_df['station_id'].unique()

    prediction_data = pd.DataFrame({
        'datetime': np.tile(next_24_hours, len(all_stations)),
        'station_id': np.repeat(all_stations, len(next_24_hours))
    })

    prediction_data['Split_hour'] = prediction_data['datetime'].dt.hour
    prediction_data['day_of_week'] = 1
    prediction_data['is_holiday'] = False
    prediction_data['feels_like'] = 3.0 
    prediction_data['Day'] = 22
    prediction_data['Month'] = 4
    prediction_data = prediction_data[['station_id', 'Split_hour', 'day_of_week', 'is_holiday', 'feels_like', 'Day', 'Month']]
    model = load('random_forest_model.joblib')
    y_pred = model.predict(prediction_data)
    prediction_data['predicted_net_flow'] = y_pred
    return prediction_data


def optimize_bike_allocation(pre_df, cost_missed, cost_excess):
    station_df = pd.read_csv('Merged_Station.csv', index_col=0)
    station_df = station_df.sort_values(by='Station_ID')
    model = pulp.LpProblem("Bike_Station_Rebalancing", pulp.LpMinimize)
    x = {i: pulp.LpVariable(f'bike_{i}', lowBound=0, cat='Integer') for i in station_df['Station_ID']}
    unmet_demand = {(i, t): pulp.LpVariable(f'unmet_demand_{i}_{t}', lowBound=0) for i in station_df['Station_ID'] for t in range(24)}
    excess_bikes = {(i, t): pulp.LpVariable(f'excess_bikes_{i}_{t}', lowBound=0) for i in station_df['Station_ID'] for t in range(24)}

    model += pulp.lpSum([
        cost_missed * unmet_demand[i, t] + cost_excess * excess_bikes[i, t] 
        for i in station_df['Station_ID'] 
        for t in range(24)
    ])

    for i in station_df['Station_ID']:
        for t in range(24):
            predicted_net_flow = pre_df.loc[(pre_df['station_id'] == i) & (pre_df['Split_hour'] == t), 'predicted_net_flow'].sum()
            model += unmet_demand[i, t] >= predicted_net_flow - x[i]
            model += excess_bikes[i, t] >= x[i] - predicted_net_flow
            model += x[i] <= station_df.loc[station_df['Station_ID'] == i, 'Total Docks'].values[0]  # Capacity constraint

    model.solve()

    put_text("Status: %s " % pulp.LpStatus[model.status])
    for station in x:

        put_text('station: %d, Allocate %d bikes' % (station, x[station].varValue))
        #put_text(f"Station {station}: Allocate {x[station].varValue} bikes")


def index():
    cost_missed = input("Cost_missed(dollar):", type=FLOAT)  # Default value is 5 if not provided
    cost_excess = input("Cost_excess(dollar):", type=FLOAT)  # Default value is 1 if not provided

    net_flows = predict_net_flow()
    optimize_bike_allocation(net_flows, cost_missed, cost_excess)
    

if __name__ == '__main__':
    #start_server(index, port=66)
    index()



