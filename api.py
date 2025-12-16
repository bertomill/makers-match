"""
FastAPI server for running the MakersLounge matcher via web interface.
Run with: uvicorn api:app --reload --port 8000
"""

import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile

# Import functions from matcher.py
from matcher import (
    parse_csv,
    run_preflight_check,
    run_all_rounds,
    generate_coverage_report,
    run_ai_validation,
    export_to_csv,
)

app = FastAPI(title="MakersLounge Matcher API")

# Allow CORS for frontend (permissive for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Store for current matching session
current_session = {
    "status": "idle",
    "step": None,
    "substep": None,
    "message": None,
    "detail": None,
    "progress": 0,  # 0-100 percentage
    "attendees": None,
    "results": None,
    "coverage": None,
    "validation": None,
}

def update_progress(step: int, message: str, detail: str = None, progress: int = 0):
    """Update the current session progress"""
    current_session["step"] = step
    current_session["message"] = message
    current_session["detail"] = detail
    current_session["progress"] = progress


@app.get("/")
def root():
    return {"status": "ok", "message": "MakersLounge Matcher API"}


@app.get("/status")
def get_status():
    """Get current matching status"""
    return current_session


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV and parse attendees"""
    global current_session

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    current_session["status"] = "uploading"
    current_session["step"] = 1
    current_session["message"] = "Uploading CSV..."

    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)

        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Parse CSV
        current_session["message"] = "Parsing attendees..."
        attendees = parse_csv(temp_path)

        # Filter to complete profiles
        complete = [a for a in attendees if a.has_complete_profile]
        incomplete = [a for a in attendees if not a.has_complete_profile]

        current_session["status"] = "parsed"
        current_session["step"] = 1
        current_session["message"] = f"Loaded {len(complete)} attendees"
        current_session["attendees"] = {
            "total": len(attendees),
            "complete": len(complete),
            "incomplete": len(incomplete),
            "incomplete_names": [a.name for a in incomplete],
            "temp_path": temp_path,
        }

        return {
            "success": True,
            "total_attendees": len(attendees),
            "complete_profiles": len(complete),
            "incomplete_profiles": len(incomplete),
            "incomplete_names": [a.name for a in incomplete],
        }

    except Exception as e:
        current_session["status"] = "error"
        current_session["message"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run")
async def run_matching():
    """Run the full matching algorithm"""
    global current_session

    if not current_session.get("attendees"):
        raise HTTPException(status_code=400, detail="No CSV uploaded. Upload a CSV first.")

    temp_path = current_session["attendees"]["temp_path"]

    try:
        # Step 1: Load attendees (already done)
        current_session["status"] = "running"
        update_progress(1, "Loading attendees...", "Parsing CSV file", 5)

        attendees = parse_csv(temp_path)
        attendees = [a for a in attendees if a.has_complete_profile]

        update_progress(1, "Loading attendees...", f"Found {len(attendees)} complete profiles", 10)

        # Step 2: Pre-flight check
        update_progress(2, "Pre-flight validation", "Checking data quality...", 15)

        preflight = run_preflight_check(attendees)

        if preflight["errors"]:
            current_session["status"] = "error"
            current_session["message"] = "Pre-flight check failed"
            return {
                "success": False,
                "step": 2,
                "errors": preflight["errors"],
            }

        update_progress(2, "Pre-flight validation", "All checks passed!", 20)

        # Step 3: Run matching rounds (this is the long part)
        # We'll run each round individually to update progress
        update_progress(3, "AI Matching", "Starting Round 1 (Complementary)...", 25)

        from matcher import run_complementary_round, run_similarity_round

        all_rounds = []
        used_pairs = set()

        # Round 1: Complementary
        update_progress(3, "AI Matching", "Round 1: Finding skill→need matches...", 30)
        round1 = run_complementary_round(1, attendees, used_pairs)
        all_rounds.append(round1)
        for m in round1["matches"]:
            pair = tuple(sorted([m["helper"]["name"], m["helped"]["name"]]))
            used_pairs.add(pair)

        # Round 2: Complementary
        update_progress(3, "AI Matching", "Round 2: Finding more skill→need matches...", 45)
        round2 = run_complementary_round(2, attendees, used_pairs)
        all_rounds.append(round2)
        for m in round2["matches"]:
            pair = tuple(sorted([m["helper"]["name"], m["helped"]["name"]]))
            used_pairs.add(pair)

        # Round 3: Similarity
        update_progress(3, "AI Matching", "Round 3: Finding similar project matches...", 60)
        round3 = run_similarity_round(3, attendees, used_pairs)
        all_rounds.append(round3)
        for m in round3["matches"]:
            pair = tuple(sorted([m["person_a"]["name"], m["person_b"]["name"]]))
            used_pairs.add(pair)

        # Round 4: Similarity
        update_progress(3, "AI Matching", "Round 4: Finding more similar projects...", 75)
        round4 = run_similarity_round(4, attendees, used_pairs)
        all_rounds.append(round4)

        results = {
            "total_attendees": len(attendees),
            "total_matches": sum(r["match_count"] for r in all_rounds),
            "rounds": all_rounds,
        }
        current_session["results"] = results

        # Step 4: Save results
        update_progress(4, "Saving results", "Writing matches.json...", 80)

        output_dir = os.path.dirname(temp_path)
        output_path = os.path.join(output_dir, 'matches.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Also copy to the web app's public folder
        web_public_path = os.path.join(
            os.path.dirname(__file__),
            'makerslounge-web', 'public', 'matches.json'
        )
        shutil.copy(output_path, web_public_path)

        update_progress(4, "Saving results", "Results saved!", 82)

        # Step 5: Export CSVs
        update_progress(5, "Exporting CSV files", "Creating round CSVs...", 85)
        export_to_csv(results, output_dir)

        # Step 6: Coverage report
        update_progress(6, "Coverage report", "Analyzing match coverage...", 90)

        coverage = generate_coverage_report(results, attendees)
        current_session["coverage"] = coverage

        update_progress(6, "Coverage report", f"{coverage['coverage_percentage']}% coverage achieved!", 92)

        # Step 7: AI Validation (optional - don't fail if this errors)
        update_progress(7, "AI Validation", "Running quality analysis...", 95)

        try:
            validation = run_ai_validation(results, attendees)
            current_session["validation"] = validation
        except Exception as e:
            print(f"AI Validation failed (non-critical): {e}")
            validation = {"overall_quality": "N/A", "issues": [], "error": str(e)}
            current_session["validation"] = validation

        # Complete!
        current_session["status"] = "complete"
        update_progress(7, "Complete!", f"Created {results['total_matches']} matches", 100)

        return {
            "success": True,
            "total_matches": results["total_matches"],
            "rounds": [
                {
                    "round": r["round"],
                    "type": r["type"],
                    "match_count": r["match_count"],
                }
                for r in results["rounds"]
            ],
            "coverage": {
                "total_attendees": coverage["total_attendees"],
                "coverage_percentage": coverage["coverage_percentage"],
                "under_matched_count": len(coverage["under_matched"]),
            },
            "validation": {
                "overall_quality": validation.get("overall_quality", "N/A"),
                "issues_count": len(validation.get("issues", [])),
            },
        }

    except Exception as e:
        current_session["status"] = "error"
        current_session["message"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results")
def get_results():
    """Get the full matching results"""
    if not current_session.get("results"):
        raise HTTPException(status_code=404, detail="No results available. Run matching first.")

    return {
        "results": current_session["results"],
        "coverage": current_session["coverage"],
        "validation": current_session["validation"],
    }


@app.post("/reset")
def reset_session():
    """Reset the current session"""
    global current_session
    current_session = {
        "status": "idle",
        "step": None,
        "message": None,
        "attendees": None,
        "results": None,
        "coverage": None,
        "validation": None,
    }
    return {"success": True, "message": "Session reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
