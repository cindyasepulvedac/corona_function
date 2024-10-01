import uvicorn
from fastapi import FastAPI, HTTPException, status, Depends, Response
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Annotated
from jose import jwt
import io
import numpy as np
import pickle
import base64
from PIL import Image

users = {
    'user1': {'username':'user1', 'email':'user1@corona.com', 'password':'123'},
    'user2': {'username':'user2', 'email':'user2@corona.com', 'password':'789'}
}

app = FastAPI()
app.title = 'Hand-written digits recognizing model'
app.version = '1.0.1'
oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')

def encode_token(payload: dict) -> str:
    token = jwt.encode(payload, 'my-secret', algorithm='HS256')
    return token

def decode_token(token: Annotated[str, Depends(oauth2_scheme)]) -> dict:
    data = jwt.decode(token, 'my-secret', algorithms=['HS256'])
    user = users.get(data['username'])
    return user

class PredictionRequest(BaseModel):
    request_id: str
    modelo: str
    image: str
    access_token: str

@app.get('/')
def greeting():
    return 'Hand-written digits recognizing model'

@app.post('/token', tags=['Authentication'])
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = users.get(form_data.username)
    if not user or form_data.password != user['password']: 
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Usuario o contraseña incorrecta'
        )
    token = encode_token({'username': user['username'], 'email': user['email']})
    
    return {'access_token': token, 'token_type': 'bearer'}


@app.post('/predict', tags=['Model prediction'])
def predict(request: PredictionRequest):
    try:
        # Validar el token
        current_user = decode_token(request.access_token)

        if request.modelo != 'clf.pickle':
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Modelo no soportado. Use "clf.pickle".'
            ) 
        
        else:

            ## Cargar modelo serializado
            with open('clf.pickle', 'rb') as f:
                clf = pickle.load(f)
            
            ## Decodificar la imagen de base64 a bytes
            try:
                bytes_image = str.encode(request.image)
            except:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='La imagen proporcionada no es una cadena base64 válida'
                )
            
            ## Abrir la imagen usando PIL
            try:
                img = Image.open(io.BytesIO(base64.decodebytes(bytes_image)))
            except:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail='No se pudo abrir la imagen. Asegúrese que sea una imagen válida'
                )
            
            ## Convertir la imagen a un NumPy array
            number = np.round((np.array(img)/255)*16)

            ## Identificación de la imagen
            prediction = clf.predict(number.reshape(1,-1))

            result = {
                    'request_id': request.request_id,
                    'prediction': prediction.tolist()
                }
        
            return JSONResponse(content=result, status_code=status.HTTP_200_OK)
           
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Se produjo un error interno: {str(e)}'
        )
