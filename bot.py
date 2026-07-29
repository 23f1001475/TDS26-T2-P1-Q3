import os
import json
import time
import logging
from datetime import datetime

import requests
import openai
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOG_PUBLIC_URL = os.getenv("LOG_PUBLIC_URL", "none")
LOCAL_LOG_PATH = os.getenv("LOCAL_LOG_PATH", "run.jsonl")

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not set. Exiting.")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. OpenAI calls will fail if attempted.")

# Configure OpenAI key only if provided
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    openai.api_key = None

PROMPT_INSTRUCTIONS = (
    "You are a careful data-analyst agent. The user will send a plain-text message asking a data-analysis question. "
    "You must reply with exactly one JSON object and nothing else. The object MUST have two keys: \"answer\" and \"log_url\". "
    "The value of \"answer\" should be shaped exactly as the user's message requests (for example, if the user asks 'Reply with ONLY this JSON object: {\"answer\": {\"state\": \"<state name>\"}, \"log_url\": \"<url>\"}', then \"answer\" must be an object with key \"state\", etc.). "
    "The value of \"log_url\" must be the public wget-able URL where the agent's run log will be available. Use the provided LOG_PUBLIC_URL value. Do not include any explanatory text or extra fields. "
    "If you need to fetch data from a URL mentioned in the user's message, do so. If the user provided inline CSV or data, parse it. Keep outputs concise and strictly follow the requested JSON shape."
)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = update.message.text.strip() if update.message and update.message.text else ""
    logger.info("Received message from %s (%s): %s", user.username, user.id, text)

    timestamp = datetime.utcnow().isoformat() + "Z"

    # Build system+user prompt for OpenAI
    system_prompt = PROMPT_INSTRUCTIONS
    # include the log URL the bot will publish
    system_prompt += f"\nLOG_PUBLIC_URL={LOG_PUBLIC_URL}\n"

    user_prompt = (
        "Here is the user's message. Follow my earlier instructions and reply with exactly one JSON object and nothing else.\n"
        "Message:\n" + text + "\n\nRespond with only the JSON object."
    )

    model_response_text = None
    model_used = None
    try:
        # Call OpenAI ChatCompletion
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1000,
        )
        model_used = resp.get("model")
        model_response_text = resp["choices"][0]["message"]["content"].strip()
        logger.info("Model reply: %s", model_response_text)
    except Exception as e:
        logger.exception("OpenAI call failed: %s", e)
        # Produce a safe JSON error response (must still be exactly one JSON object)
        fallback = {"answer": {"error": "openai_call_failed", "message": str(e)}, "log_url": LOG_PUBLIC_URL}
        model_response_text = json.dumps(fallback, ensure_ascii=False)

    # Ensure the model output is a single JSON object
    parsed = None
    try:
        parsed = json.loads(model_response_text)
        if not isinstance(parsed, dict):
            raise ValueError("Top-level JSON is not an object")
    except Exception as e:
        logger.warning("Model output not valid JSON object: %s", e)
        # Try a repair pass: ask the model to extract the JSON only
        try:
            repair_prompt = (
                "The previous model output was not a valid single JSON object. Extract or produce the exact single JSON object required (with keys 'answer' and 'log_url') and nothing else. "
                "User message was:\n" + text + "\nPrevious model output:\n" + model_response_text
            )
            resp2 = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[{"role": "system", "content": PROMPT_INSTRUCTIONS}, {"role": "user", "content": repair_prompt}],
                temperature=0.0,
                max_tokens=800,
            )
            model_response_text = resp2["choices"][0]["message"]["content"].strip()
            parsed = json.loads(model_response_text)
            if not isinstance(parsed, dict):
                raise ValueError("Repaired output not an object")
        except Exception as e2:
            logger.exception("Repair failed: %s", e2)
            # Final fallback JSON
            parsed = {"answer": {"error": "could_not_produce_valid_json"}, "log_url": LOG_PUBLIC_URL}
            model_response_text = json.dumps(parsed, ensure_ascii=False)

    # Ensure log_url in parsed is set to LOG_PUBLIC_URL
    try:
        parsed["log_url"] = LOG_PUBLIC_URL
    except Exception:
        parsed = {"answer": {"error": "invalid_parsed_structure"}, "log_url": LOG_PUBLIC_URL}
        model_response_text = json.dumps(parsed, ensure_ascii=False)

    # Write run log locally (append JSONL)
    run_entry = {
        "timestamp": timestamp,
        "user_id": user.id,
        "username": user.username,
        "message": text,
        "model": model_used,
        "response_text": model_response_text,
        "parsed_response": parsed,
    }
    try:
        with open(LOCAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Failed to write local log")

    # Optionally, try to upload the log file to a user-provided upload endpoint
    # If LOG_UPLOAD_ENDPOINT is set, attempt a PUT and log result (not required)
    log_upload_endpoint = os.getenv("LOG_UPLOAD_ENDPOINT")
    if log_upload_endpoint:
        try:
            with open(LOCAL_LOG_PATH, "rb") as f:
                r = requests.put(log_upload_endpoint, data=f)
            logger.info("Uploaded log to %s status=%s", log_upload_endpoint, r.status_code)
        except Exception:
            logger.exception("Log upload failed")

    # Send exactly the JSON object as the reply text
    reply_text = json.dumps(parsed, ensure_ascii=False)
    await update.message.reply_text(reply_text)


def main():
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN environment variable required.")
        return

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    app.add_handler(handler)

    print("Bot starting. Listening for messages...")
    app.run_polling()


if __name__ == "__main__":
    main()
