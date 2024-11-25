from fastapi import APIRouter, Request
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from services.llm_service import process_message
from utils.config import TELEGRAM_TOKEN
import logging

logging.basicConfig(level=logging.INFO)

router = APIRouter()

telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    chat_id = update.effective_chat.id
    logging.info(f"Mensaje recibido de {chat_id}: {user_message}")
    response = await process_message(user_message)
    logging.info(f"Mensaje enviado {response}")
    await context.bot.send_message(chat_id=chat_id, text=response)

telegram_app.add_handler(MessageHandler(filters.TEXT &  ~filters.COMMAND, message_handler))

@router.post("/telegram/webhook")
async def telegram_webkook(request:Request):
    update = Update.de_json(await request.json(), telegram_app.bot)
    await telegram_app.process_update(update)
    return "OK"