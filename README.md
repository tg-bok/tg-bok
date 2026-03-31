import logging
import random
import os
from datetime import datetime
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

# ======= 配置 =======
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
HISTORY_FILE = "users_history.txt"
GROUP_FILE = "groups.txt"

# ======= 矿池基础数据 =======
users_base = 863351
total_amount_base = 873508118
total_profit_base = 324737541

users_daily_growth = 85
total_amount_daily_growth = 1850608
total_profit_daily_growth = 35687

daily_profit_fixed = 2.8

scheduled_times = ["10:00", "15:00", "18:00"]

logging.basicConfig(level=logging.INFO)

# ====== 群 / 频道记录 ======
def load_groups():
    if os.path.exists(GROUP_FILE):
        with open(GROUP_FILE, "r") as f:
            return list(set([line.strip() for line in f.readlines()]))
    return []

def save_group(chat_id):
    groups = load_groups()
    if str(chat_id) not in groups:
        with open(GROUP_FILE, "a") as f:
            f.write(str(chat_id) + "\n")

# ====== 用户历史 ======
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            users = [line.strip().split("|") for line in f.readlines()]
            return [{"address": u[0], "balance": int(u[1])} for u in users]
    return []

def save_history(users):
    with open(HISTORY_FILE, "w") as f:
        for u in users:
            f.write(f"{u['address']}|{u['balance']}\n")

# ====== 生成用户 ======
def generate_new_user():
    addr = "0x" + ''.join(random.choices("ABCDEF0123456789", k=6)) + "***" + ''.join(random.choices("ABCDEF0123456789", k=2))

    tier = random.choice(["low", "mid", "high"])
    if tier == "low":
        balance = random.randint(300, 1500)
    elif tier == "mid":
        balance = random.randint(2000, 10000)
    else:
        balance = random.randint(15000, 80000)

    return {"address": addr, "balance": balance}

def get_latest_users(n=10):
    history = load_history()
    selected = []
    available = history.copy()

    while len(selected) < n:
        if available:
            u = random.choice(available)
            selected.append(u)
            available.remove(u)
        else:
            new_user = generate_new_user()
            history.append(new_user)
            selected.append(new_user)

    save_history(history)
    return selected

# ====== 矿池数据 ======
def get_current_pool_values():
    tz = pytz.timezone('America/New_York')
    now = datetime.now(tz)
    seconds_today = (now - datetime.combine(now.date(), datetime.min.time(), tzinfo=tz)).total_seconds()

    users_now = users_base + int(users_daily_growth * seconds_today / 86400)
    total_amount_now = total_amount_base + int(total_amount_daily_growth * seconds_today / 86400)
    total_profit_now = total_profit_base + int(total_profit_daily_growth * seconds_today / 86400)

    return users_now, total_amount_now, total_profit_now

# ====== 生成消息 ======
def generate_pool_message():
    users_now, total_amount_now, total_profit_now = get_current_pool_values()

    message = "🔥 OnchainETH Node 🔥\n\n"
    message += f"💰 Total Pool: {total_amount_now:,} USDT\n"
    message += f"👥 Participants: {users_now:,}\n"
    message += f"📈 Daily Profit: {daily_profit_fixed}%\n"
    message += f"💹 Total Profit: {total_profit_now:,} USDT\n\n"

    message += "🟢 User Details:\n"

    latest_users = get_latest_users(10)

    for u in latest_users:
        balance = u["balance"]
        profit_today = int(balance * daily_profit_fixed / 100)
        message += f"{u['address']} — Balance: {balance:,} USDT — Profit Today: {profit_today:,} USDT\n"

    message += "\n⚠️ Randomly selected user data."

    return message

# ====== 自动记录群 / 频道 ======
async def track_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat

    if chat.type in ["group", "supergroup", "channel"]:
        save_group(chat.id)

# ====== 命令 ======
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot activated ✅")
    await update.message.reply_text(generate_pool_message())

async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(generate_pool_message())

async def latest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = "🟢 User Details:\n"
    users = get_latest_users(10)

    for u in users:
        balance = u["balance"]
        profit_today = int(balance * daily_profit_fixed / 100)
        msg += f"{u['address']} — Balance: {balance:,} USDT — Profit Today: {profit_today:,} USDT\n"

    msg += "\n⚠️ Randomly selected user data."

    await update.message.reply_text(msg)

# ====== 定时群发 ======
async def send_pool_info(bot: Bot):
    msg = generate_pool_message()
    groups = load_groups()

    for group_id in groups:
        try:
            await bot.send_message(chat_id=group_id, text=msg)
        except Exception as e:
            logging.error(f"Send failed: {e}")

def schedule_jobs(application):
    tz = pytz.timezone('America/New_York')
    scheduler = AsyncIOScheduler(timezone=tz)

    for t in scheduled_times:
        hour, minute = map(int, t.split(":"))
        scheduler.add_job(
            lambda: application.bot.loop.create_task(send_pool_info(application.bot)),
            'cron',
            hour=hour,
            minute=minute
        )

    scheduler.start()

# ====== 启动 ======
async def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("info", info))
    app.add_handler(CommandHandler("latest", latest))

    app.add_handler(MessageHandler(filters.ALL, track_group))

    schedule_jobs(app)

    await app.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

