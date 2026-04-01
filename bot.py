import os
import random
import asyncio
import pytz
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from apscheduler.schedulers.background import BackgroundScheduler

BOT_TOKEN = os.environ.get("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not set")

app = ApplicationBuilder().token(BOT_TOKEN).build()
groups = set()

async def auto_save_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat:
        return
    chat = update.effective_chat
    if chat.type in ["group", "supergroup"]:
        if chat.id not in groups:
            groups.add(chat.id)
            print(f"Saved group: {chat.id}")

def get_rate(balance):
    if 100 <= balance <= 999:
        return 0.01
    elif 1000 <= balance <= 4999:
        return 0.015
    elif 5000 <= balance <= 9999:
        return 0.02
    elif 10000 <= balance <= 49999:
        return 0.025
    elif 50000 <= balance <= 99999:
        return 0.03
    elif 100000 <= balance <= 299999:
        return 0.035
    elif 300000 <= balance <= 499999:
        return 0.04
    elif 500000 <= balance <= 1000000:
        return 0.05
    else:
        return 0.01

def generate_users(count):
    users = []
    for i in range(count):
        balance = random.randint(100, 20000)
        if random.random() < 0.15:
            balance = random.randint(50000, 200000)
        if random.random() < 0.05:
            balance = random.randint(300000, 800000)
        balance = int(balance * random.uniform(0.95, 1.05))
        users.append({"name": f"User{i+1}", "balance": balance})
    return users

async def generate_pool_message(chat_id, context):
    total_users = await context.bot.get_chat_member_count(chat_id)
    users = generate_users(total_users)
    total_funds = sum(u["balance"] for u in users)
    total_profit = sum(u["balance"] * get_rate(u["balance"]) for u in users)

    msg = "<pre>"
    msg += "Onchain Wallet ETH Mining Node\n\n"
    msg += f"Total Users: {total_users}\n"
    msg += f"Total Funds: ${total_funds}\n"
    msg += "Daily Rate: 1% - 5%\n"
    msg += f"Total Profit Today: ${total_profit:.2f}\n\n"

    selected_users = random.sample(users, min(10, len(users)))
    msg += "User Details:\n"
    for u in selected_users:
        rate = get_rate(u["balance"])
        profit = u["balance"] * rate
        msg += f"{u['name']}: Balance ${u['balance']}, Rate {rate*100:.1f}%, Profit ${profit:.2f}\n"

    msg += "\nReal-time onchain data update"
    msg += "</pre>"
    return msg

async def send_pool_message():
    if not groups:
        print("No groups found")
        return
    for chat_id in groups:
        try:
            text = await generate_pool_message(chat_id, app)
            await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
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
