from os import getenv
import requests
from functools import reduce
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from dotenv import load_dotenv

load_dotenv()

STT_BASE_URL = getenv('STT_BASE_URL') or "https://sn-watson-stt.labs.skills.network"
TTS_BASE_URL = getenv('TTS_BASE_URL') or "https://sn-watson-tts.labs.skills.network"
Watsonx_API = getenv('IBM_WATSON_X_API_KEY')
Project_id = getenv('IBM_WATSON_X_PROJECT_ID') or 'skills-network'
Watsonx_URL = getenv('IBM_WATSON_X_URL') or 'https://us-south.ml.cloud.ibm.com'

credentials = {
    'url': Watsonx_URL,
    'apikey' : Watsonx_API
}

model_id = ModelTypes.FLAN_UL2

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}

model = Model(
    model_id = model_id,
    params = parameters,
    credentials = credentials,
    project_id = Project_id
)

def speech_to_text(audio_binary):
    api_url = f'{STT_BASE_URL}/speech-to-text/api/v1/recognize'

    params = { 'model': 'en-US_Multimedia' }

    response = requests.post(api_url, params=params, data=audio_binary).json()

    text = 'null'
    if not bool(response.get('results')):
        return text

    flattened_result = reduce(lambda accum, curr: accum + curr.get('alternatives', []), response.get('results'), [])

    winning_prediction = max(flattened_result, key=lambda r: r.get('confidence'))

    text = winning_prediction.get('transcript', text).strip()

    return text

def text_to_speech(text, voice=""):
    api_url = f'{TTS_BASE_URL}/text-to-speech/api/v1/synthesize?output=output_text.wav'

    if voice and voice != 'default':
        api_url = f'{api_url}&voice={voice}'

    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json'
    }

    data = {
        'text': text
    }

    response = requests.post(api_url, headers=headers, json=data)
    print('Text to Speech response: ', response)

    return response.content

def watsonx_process_message(user_message):
    prompt = f'''Respond to the query: ```{user_message}```'''
    response_text = model.generate_text(prompt=prompt)
    print('WatsonX response: ', response_text)

    return response_text

