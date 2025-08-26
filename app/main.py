from pathlib import Path
import sys
import os
from flask import Flask, render_template, request, jsonify, make_response


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


generated_dir = PROJECT_ROOT / "static" / "generated"
generated_dir.mkdir(parents=True, exist_ok=True)


from app.modules.llm_provider import make_llm
from app.modules.agent import DataframeAgent
from app.modules.utils import load_csv_to_df, set_session, get_session, update_session


templates_dir = PROJECT_ROOT / "templates"
static_dir = PROJECT_ROOT / "static"

app = Flask(
    __name__,
    template_folder=str(templates_dir),
    static_folder=str(static_dir),
)

app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB uploads

@app.get('/')
def index():
    sid = request.cookies.get("sid")
    if not sid:
        sid = set_session({})
        resp = make_response(render_template('index.html'))
        resp.set_cookie("sid", sid, httponly=True, samesite='Lax')
        return resp
    else:
        return render_template('index.html')

@app.post('/setup')
def setup():
    sid = request.cookies.get("sid")
    if not sid:
        sid = set_session({})
    groq_key = request.form.get("groq_api_key", "").strip()
    model = request.form.get("model", "llama-3-3-70b-instruct").strip() or "llama-3-3-70b-instruct"
    file = request.files.get("csv_file")
    if not groq_key:
        return jsonify({"ok": False, "error": "Groq API key is required."}), 400
    if not file:
        return jsonify({"ok": False, "error": "CSV file is required."}), 400

    try:
        df = load_csv_to_df(file)
        llm = make_llm(groq_api_key=groq_key, model=model)
        agent = DataframeAgent(llm, df)
        update_session(sid, agent=agent)  # keep in memory only
        return jsonify({"ok": True, "message": "Agent is ready."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post('/chat')
def chat():
    sid = request.cookies.get("sid")
    store = get_session(sid)
    agent = store.get("agent")
    if not agent:
        return jsonify({"ok": False, "error": "Please provide API key and CSV first."}), 400
    user_msg = (request.json or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"ok": False, "error": "Empty message."}), 400
    result = agent.ask(user_msg)
    return jsonify(result), (200 if result.get("ok") else 500)

if __name__ == "__main__":
    print(f"Starting app from: {THIS_FILE}")
    print(f"Project root: {PROJECT_ROOT}")
    print("Templates folder:", templates_dir)
    print("Static folder:", static_dir)
    app.run(host="0.0.0.0", port=7860, debug=True)
