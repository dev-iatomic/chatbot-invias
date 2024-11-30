import os
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, Request
import requests
import json
from services.llm_service import process_message  # Importar la función process_message

app = FastAPI()

# Definir el token del bot y la URL del webhook
TOKEN = ""
WEBHOOK_URL = "/webhook"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TOKEN}"

# Configurar el webhook
@app.on_event("startup")
async def set_webhook():
    url = f"{TELEGRAM_API_URL}/setWebhook?url={WEBHOOK_URL}"
    response = requests.get(url)
    if response.status_code == 200:
        print("Webhook configurado correctamente")
    else:
        print(f"Error configurando el webhook: {response.content}")

# Definir el endpoint del webhook
@app.post("/webhook")
async def webhook(request: Request):
    # Obtener la data enviada por Telegram
    data = await request.json()
    
    # Extraer el mensaje recibido y el chat ID
    message = data.get("message")
    if message:
        chat_id = message["chat"]["id"]
        text = message.get("text", "")

        # Procesar el mensaje usando la función process_message
        try:
            response_text = await process_message(text)
        except Exception as e:
            response_text = "Hubo un error al procesar tu mensaje, por favor intenta nuevamente."

        # Enviar la respuesta procesada al chat
        reply_url = f"{TELEGRAM_API_URL}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": response_text
        }

        headers = {"Content-Type": "application/json"}
        requests.post(reply_url, data=json.dumps(payload), headers=headers)

    return {"status": "ok"}
