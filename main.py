import os
import random
from datetime import datetime
import pytz
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from apscheduler.schedulers.background import BackgroundScheduler


BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")  # 


app = ApplicationBuilder().token(BOT_TOKEN).build()


USERS = [
    {"name": "Alice", "balance": 5000},
    {"name": "Bob", "balance": 15000},
    {"name": "Charlie", "balance": 700},
    {"name": "David", "balance": 12000},
    {"name": "Eva", "balance": 300},
    {"name": "Frank", "balance": 2500},
    {"name": "Grace", "balance": 8000},
    {"name": "Hank", "balance": 600},
    {"name": "Ivy", "balance": 4000},
    {"name": "Jack", "balance": 900},
    {"name": "Kate", "balance": 10000},
    {"name": "Leo", "balance": 3500},
]

DAILY_RATE = 0.02  2%，


def generate_pool_message():
    
    selected_users = random.sample(USERS, 10)
    
    msg = "💎 OnchainETH Node\n"
    total_users = len(USERS)
    total_funds = sum(u["balance"] for u in USERS)
    total_profit = sum(u["balance"] * DAILY_RATE for u in USERS)
    
    msg += f"Total Users: {total_users}\n"
    msg += f"Total Funds: ${total_funds}\n"
    msg += f"Daily Rate: {DAILY_RATE*100}%\n"
    msg += f"Total Profit Today: ${total_profit:.2f}\n\n"
    
    msg += "🔹 User Details:\n"
    for u in selected_users:
        profit = u["balance"] * DAILY_RATE
        msg += f"{u['name']}: Balance ${u['balance']}, Profit Today ${profit:.2f}\n"
    
    msg += "\nRandomly selected partial user details."
    return msg

async def send_pool_message():
    if CHAT_ID:
        await app.bot.send_message(chat_id=int(CHAT_ID), text=generate_pool_message())
    else:
        print("CHAT_ID not set, skipping send_pool_message")

# /start 命令
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot is running!")

app.add_handler(CommandHandler("start", start))


scheduler = BackgroundScheduler(timezone=pytz.timezone("America/New_York"))
scheduler.add_job(lambda: app.create_task(send_pool_message()), 'cron', hour=10, minute=0)
scheduler.add_job(lambda: app.create_task(send_pool_message()), 'cron', hour=15, minute=0)
scheduler.add_job(lambda: app.create_task(send_pool_message()), 'cron', hour=18, minute=0)
scheduler.start()


def main():
    print("Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
