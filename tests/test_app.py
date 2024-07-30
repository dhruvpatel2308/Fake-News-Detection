import pytest
from app import app

@pytest.fixture
def client():
  with app.test_client() as client:
    yield client

def test_home(client):
  rv = client.get('/')
  assert rv.status_code == 200
  assert b'Fake News Detection' in rv.data

def test_predict_logistic(client):
  rv = client.post('/predict', data = dict(news_text = "This is a test news article", model_choice="logistic"))
  assert rv.status_code == 200
  json_data = rv.get_json()
  assert 'prediction' in json_data
  assert 'probability' in json_data

def test_predict_bert(client):
  rv = client.post('/predict', data= dict(news_text = "This is a test news article", model_choice ="bert"))
  assert rv.status_code == 200
  json_data = rv.get_json()
  assert 'prediction' in json_data
  assert 'probability' in json_data
