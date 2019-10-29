import os
import tempfile
import pytest
import flask
import json

import server

"""
Pytest unit tests for server rest API
"""

# Unit test will fail unless correct model features added
MODEL_FEATURES = {'Unnamed: 0': 288157, 'Urban Rural_label': 'Urban', 'Carriageway_Hazards_label': 'None', 'Special_Conditions_at_Site_label': 'None', 'Road Surface_label': 'Wet or damp', 'Weather_label': 'Raining no high winds', 'Light_Conditions_label': 'Daylight', 'Ped Cross - Physical_label': 'No physical crossing facilities within 50 metres', 'Ped Cross - Human_label': 'None within 50 metres ', '2nd_Road_Class_label': 'Unclassified', 'Junction_Control_label': 'Give way or uncontrolled', 'Junction_Detail_label': 'Private drive or entrance', 'Road_Type_label': 'Dual carriageway', '1st_Road_Class_label': 'A', 'Local_Authority_(Highway)_label': 1673, 'Local_Authority_(District)_label': 1673, 'Day_of_Week_label': 'Thursday', 'Police_Force_label': 'South Wales', 'Location_Easting_OSGR': 315622, 'Location_Northing_OSGR': 177788, 'Longitude': -3.216822, 'Latitude': 51.492669, 'Number_of_Vehicles': 3, 'Number_of_Casualties': 4, 'Date': Timestamp('2014-01-16 00:00:00'), 'Time': Timestamp('2019-10-28 16:23:00'), '1st_Road_Number': 4119, 'Speed_limit': 30, '2nd_Road_Number': 0, 'LSOA_of_Accident_Location': 8, 'year': 2014, 'len_Local_Authority_(Highway)_label': 7, 'len_Local_Authority_(District)_label': 7, 'len_LSOA_of_Accident_Location': 9.0}


INVALID_FEATURES = {"incorect_col": 0.0}


@pytest.fixture
def client():
    yield server.app.test_client()


def test_base(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.data == b"Welcome to my API"


def test_get_invalid_score(client):
    response = client.post(
        "/score",
        data=json.dumps(INVALID_FEATURES)
    )
    assert response.status_code == 500


def test_get_valid_score(client):
    response = client.post(
        "/score",
        data=json.dumps(MODEL_FEATURES)
    )

    assert response.status_code == 200
    payload = json.loads(response.data)
    assert payload["model_score"] == pytest.approx(0.97060014)
