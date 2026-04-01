import random
from datetime import datetime
from telegram import Bot
from telegram.ext import Updater
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import os


BOT_TOKEN = os.environ.get("BOT_TOKEN")  # Railway 
CHAT_ID = None  


base_users = 863351
base_funds = 873508118
base_profit = 324737541

daily_user_increase = 85
daily_funds_increase = 1850608
daily_profit_increase = 35687

user_count = base_users
total_funds = base_funds
total_profit = base_profit

daily_rate = "1%-5%"  


def generate_users(num=10):
    users = []
    for i in range(num):
        address = "0x" + "".join(random.choices("ABCDEF0123456789", k=8))
        balance = random.choice([random.randint(100, 1000),
                                 random.randint(1000, 10000),
                                 random.randint(10000, 100000)])
        profit = round(balance * random.uniform(0.01, 0.05), 2)
        users.append({"address": address, "balance": balance, "profit": profit})
    return users


def generate_pool_message():
    global user_count, total_funds, total_profit

    
    user_count += daily_user_increase / 3
    total_funds += daily_funds_increase / 3
    total_profit += daily_profit_increase / 3

    message = f"🟢 OnchainETH Node\n\n" \
              f"💰 Total Users: {int(user_count)}\n" \
              f"💵 Total Funds (USDT): {int(total_funds)}\n" \
              f"📈 Daily Rate: {daily_rate}\n" \
              f"💹 Total Profit (USDT): {int(total_profit)}\n\n" \
              f"🔹 User Details:\n"

    users = generate_users(10)
    for u in users:
        message += f"{u['address']} - Balance: {u['balance']} USDT, Profit Today: {u['profit']} USDT\n"

    message += "\nℹ️ Randomly selected partial user details."
    return message


def send_pool_message(context=None):
    bot = Bot(BOT_TOKEN)
    global CHAT_ID
    if not CHAT_ID:
        updates = bot.get_updates()
        if updates:
            CHAT_ID = updates[-1].message.chat_id
        else:
            return
    message = generate_pool_message()
    bot.send_message(chat_id=CHAT_ID, text=message)


def main():
    updater = Updater(BOT_TOKEN)
    scheduler = BackgroundScheduler(timezone=pytz.timezone("America/New_York"))

    
    scheduler.add_job(send_pool_message, "cron", hour=10, minute=0)
    scheduler.add_job(send_pool_message, "cron", hour=15, minute=0)
    scheduler.add_job(send_pool_message, "cron", hour=18, minute=0)

    scheduler.start()
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
