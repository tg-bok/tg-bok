import random
import asyncio
import json
import pytz

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from apscheduler.schedulers.background import BackgroundScheduler

BOT_TOKEN = "YOUR_BOT_TOKEN"  # replace with your bot token

app = ApplicationBuilder().token(BOT_TOKEN).build()
GROUP_FILE = "groups.json"

def load_groups():
    try:
        with open(GROUP_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_groups(groups):
    with open(GROUP_FILE, "w") as f:
        json.dump(groups, f)

async def auto_save_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat:
        return
    chat = update.effective_chat
    if chat.type in ["group", "supergroup"]:
        groups = load_groups()
        if chat.id not in groups:
            groups.append(chat.id)
            save_groups(groups)
            print(f"Saved group: {chat.id}")

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

DAILY_RATE = 0.02

def generate_pool_message():
    selected_users = random.sample(USERS, min(10, len(USERS)))
    total_users = len(USERS)
    total_funds = sum(u["balance"] for u in USERS)
    total_profit = sum(u["balance"] * DAILY_RATE for u in USERS)
    msg = "OnchainETH Node\n\n"
    msg += f"Total Users: {total_users}\n"
    msg += f"Total Funds: ${total_funds}\n"
    msg += f"Daily Rate: {DAILY_RATE*100}%\n"
    msg += f"Total Profit Today: ${total_profit:.2f}\n\n"
    msg += "User Details:\n"
    for u in selected_users:
        profit = u["balance"] * DAILY_RATE
        msg += f"{u['name']}: Balance ${u['balance']}, Profit ${profit:.2f}\n"
    msg += "\nRandom partial user data"
    return msg

async def send_pool_message():
    groups = load_groups()
    if not groups:
        print("No groups found")
        return
    for chat_id in groups:
        try:
            await app.bot.send_message(chat_id=chat_id, text=generate_pool_message())
            print(f"Sent to: {chat_id}")
        except Exception as e:
            print(f"Failed to send to {chat_id}: {e}")

def run_async_job():
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(send_pool_message())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(send_pool_message())

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot is running")

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.ALL, auto_save_group))

scheduler = BackgroundScheduler(timezone=pytz.timezone("America/New_York"))
scheduler.add_job(run_async_job, 'cron', hour=10, minute=0)
scheduler.add_job(run_async_job, 'cron', hour=15, minute=0)
scheduler.add_job(run_async_job, 'cron', hour=18, minute=0)
scheduler.start()

def main():
    print("Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
