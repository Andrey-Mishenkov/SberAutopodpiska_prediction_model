#------------------------------------------------------------------------------------------
# запустить из командной строки PowerShell
#
# E:\Projects\Python\SkillBox\final_work
# .\.venv\Scripts\activate
# uvicorn main_sber:app --reload
#------------------------------------------------------------------------------------------
# import json
# from pathlib import Path
# import os
# import glob
import dill
import pandas as pd
from datetime import datetime
import logging

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

logging.basicConfig(
    level=logging.INFO  # Уровень логирования (INFO, ERROR, DEBUG)
# Настройка логирования в файл
#     filename='app.log',  # Имя файла
#     format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# with open('models/model_prediction_sber_(JN).pkl', 'rb') as file:
with open('models/model_prediction_sber.pkl', 'rb') as file:
    model = dill.load(file)

class SessionInfo(BaseModel):
    session_id: Optional[str] = None
    client_id: Optional[str] = None
    visit_date: Optional[str] = None
    visit_time: Optional[str] = None
    visit_number: Optional[int] = None
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_adcontent: Optional[str] = None
    utm_keyword: Optional[str] = None
    device_category: Optional[str] = None
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_model: Optional[str] = None
    device_screen_resolution: Optional[str] = None
    device_browser: Optional[str] = None
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None
    target_event_action: Optional[int] = None


class Prediction(BaseModel):
    session_id: str
    pred: int
    target_action: int


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(session_info: SessionInfo):

    try:
        df = pd.DataFrame.from_dict([session_info.dict()])
        y = model['model'].predict(df)

        return {
            'session_id':    session_info.session_id,
            'pred':          int(y[0]),
            'target_action': session_info.target_event_action
        }

        logger.info(f"Response: {response}")
        return response

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

#------------------------------------------------------------------------------------------
# запустить из командной строки PowerShell
# E:\Projects\Python\SkillBox\final_work
# .\.venv\Scripts\activate
# uvicorn main_sber:app --reload
#------------------------------------------------------------------------------------------

# if __name__ == '__main__':
#     # main()
