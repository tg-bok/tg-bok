from telegram import Bot
from telegram.ext import Updater
from apscheduler.schedulers.background import BackgroundScheduler
import random
import pytz
from datetime import datetime


BOT_TOKEN = "8735949838:AAEd1sToTiP7I4tuPsU7C2fSLkfbzwymK1E"

bot = Bot(token=BOT_TOKEN)
updater = Updater(token=BOT_TOKEN, use_context=True)


total_users = 863351
total_amount = 873508118
total_profit = 324737541


def generate_users():
    users = []
    for i in range(10):
        balance = random.choice([random.randint(100, 900), random.randint(1000, 9000), random.randint(10000, 90000)])
        profit = round(balance * random.uniform(0.01, 0.05), 2)
        users.append({
            "address": f"0x{random.randint(100000,999999)}abcd{i}",
            "balance": balance,
            "profit": profit
        })
    return users


def send_pool_message(context):
    global total_users, total_amount, total_profit
    total_users += 85
    total_amount += 1850608
    total_profit += 35687
    users = generate_users()
    
    msg = f"OnchainETH Node\n\n"
    msg += f"Total Users: {total_users}\n"
    msg += f"Total Amount: ${total_amount}\n"
    msg += f"Daily Interest Rate: 1%-5%\n"
    msg += f"Total Profit: ${total_profit}\n\n"
    msg += "User Details:\n"
    for u in users:
        msg += f"{u['address']} - Balance: ${u['balance']} - Profit Today: ${u['profit']}\n"
    msg += "\nRandomly selected user details."
    
    
    chat_id = "YOUR_CHAT_ID"
    context.bot.send_message(chat_id=chat_id, text=msg)


scheduler = BackgroundScheduler(timezone=pytz.timezone("America/New_York"))
scheduler.add_job(send_pool_message, 'cron', hour=10, minute=0, args=[updater.job_queue])
scheduler.add_job(send_pool_message, 'cron', hour=15, minute=0, args=[updater.job_queue])
scheduler.add_job(send_pool_message, 'cron', hour=18, minute=0, args=[updater.job_queue])
scheduler.start()


updater.start_polling()
updater.idle()
