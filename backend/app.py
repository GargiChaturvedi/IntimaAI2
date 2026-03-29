"""
Medical Coding Agent — Flask Backend
Bridges the frontend UI to the fine-tuned Bedrock model
Run with: python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add bedrock folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bedrock', 'scripts'))
from invoke_model import invoke_medical_coder, invoke_full_coding_analysis

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API


# ─── Health Check ─────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "medical-coding-agent"})


# ─── ICD-10 Code Assignment ───────────────────────────────────────────────
@app.route("/api/icd10", methods=["POST"])
def get_icd10_codes():
    """
    Request body: { "clinical_note": "..." }
    Returns: { "task": "icd10", "result": "..." }
    """
    data = request.get_json()
    if not data or "clinical_note" not in data:
        return jsonify({"error": "clinical_note is required"}), 400

    result = invoke_medical_coder(
        task="icd10",
        content=data["clinical_note"]
    )
    return jsonify(result)


# ─── CPT Code Assignment ──────────────────────────────────────────────────
@app.route("/api/cpt", methods=["POST"])
def get_cpt_codes():
    """
    Request body: { "clinical_note": "..." }
    Returns: { "task": "cpt", "result": "..." }
    """
    data = request.get_json()
    if not data or "clinical_note" not in data:
        return jsonify({"error": "clinical_note is required"}), 400

    result = invoke_medical_coder(
        task="cpt",
        content=data["clinical_note"]
    )
    return jsonify(result)


# ─── Payer Policy Check ───────────────────────────────────────────────────
@app.route("/api/payer-policy", methods=["POST"])
def check_payer_policy():
    """
    Request body: { "clinical_note": "...", "payer": "Aetna" }
    Returns: { "task": "payer_policy", "result": "..." }
    """
    data = request.get_json()
    if not data or "clinical_note" not in data:
        return jsonify({"error": "clinical_note is required"}), 400

    result = invoke_medical_coder(
        task="payer_policy",
        content=data["clinical_note"],
        payer=data.get("payer")
    )
    return jsonify(result)


# ─── Full Analysis (all three tasks in one call) ──────────────────────────
@app.route("/api/analyze", methods=["POST"])
def full_analysis():
    """
    Request body: { "clinical_note": "...", "payer": "Aetna" (optional) }
    Returns: { "task": "full_analysis", "result": "..." }
    """
    data = request.get_json()
    if not data or "clinical_note" not in data:
        return jsonify({"error": "clinical_note is required"}), 400

    result = invoke_full_coding_analysis(
        clinical_note=data["clinical_note"],
        payer=data.get("payer")
    )
    return jsonify(result)


# ─── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🏥 Medical Coding Agent backend running at http://localhost:5000")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)