"""
================================================================================
VENN MATCHING ALGORITHM - MakersLounge Speed Networking
================================================================================

PURPOSE:
    This script automatically matches attendees for 4 rounds of speed networking
    at MakersLounge events. Instead of random pairing, it uses AI to create
    meaningful connections based on what each person can offer and what they need.

HOW IT WORKS:
    1. Load the guest list from Luma (CSV export)
    2. For each attendee, extract their:
       - Superpowers (skills they can share with others)
       - Needs (what they're looking for help with)
       - Project (what they're working on)
       - Phase (Spark, Build, Momentum, or Maintenance)

    3. Run 4 rounds of matching:
       - Rounds 1-3: "Complementary" matching
         → Pair someone who HAS a skill with someone who NEEDS that skill
         → One person is the "helper", one is being "helped"

       - Round 4: "Similarity" matching
         → Pair people working on SIMILAR projects
         → They meet as equals to share experiences and ideas

    4. Output a JSON file with all matches and AI-generated explanations

CONSTRAINTS:
    - No one meets the same person twice across all 4 rounds
    - Everyone should get roughly equal time as helper vs helped
    - Maximize match quality (how well skills align with needs)

USAGE:
    python matcher.py "path/to/luma_export.csv"

OUTPUT:
    Creates a JSON file with the same name + "_matches.json"
    Example: "MakersLounge Meetup - Guests.csv" → "MakersLounge Meetup - Guests_matches.json"

================================================================================
"""

# ============================================================================
# IMPORTS - Libraries we need to make this work
# ============================================================================

import csv          # For reading the Luma CSV export
import json         # For saving matches as JSON
import random       # For shuffling attendees (ensures fairness)
import os           # For file path handling
from dataclasses import dataclass   # For clean data structures
from typing import Optional         # For type hints
from openai import OpenAI           # For AI-powered matching decisions

# For optimal bipartite matching (Hungarian algorithm)
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not installed. Using greedy matching instead of optimal.")
    print("Install with: pip install scipy")


# ============================================================================
# CONFIGURATION - Load API credentials
# ============================================================================
#
# The AI matching requires an OpenAI API key. This is stored in a .env file
# in the project root for security (never commit API keys to git!).
#
# The .env file should contain:
#   OPENAI_API_KEY=sk-proj-xxxxx
#
# ============================================================================

def load_env():
    """
    Load environment variables from the .env file.
    This keeps our API key secure and out of the code.
    """
    # Try current directory first, then parent directory
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '.env'),
        os.path.join(os.path.dirname(__file__), '..', '.env'),
    ]
    for env_path in possible_paths:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
            break

load_env()

# Initialize the OpenAI client - this is what we use to call the AI
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))


# ============================================================================
# DATA STRUCTURES - How we represent attendees and matches
# ============================================================================

@dataclass
class Attendee:
    """
    Represents one person attending the event.

    This is like a digital "profile card" for each attendee containing:
    - Their basic info (name, email, LinkedIn)
    - What they're good at (superpowers/skills)
    - What they need help with (needs)
    - What they're building (project) and how far along they are (phase)

    We also track how many times they've been a helper vs helped,
    so we can balance roles across the event.
    """
    id: str                     # Unique identifier from Luma
    name: str                   # Display name
    email: str                  # Contact email
    linkedin: str               # LinkedIn profile URL
    project: str                # What project(s) they're working on
    phase: str                  # Spark / Build / Momentum / Maintenance
    superpowers: list[str]      # Skills they can OFFER to others
    needs: list[str]            # What they NEED help with

    # Role tracking - updated as we create matches
    helper_count: int = 0       # How many times they've been the helper
    helped_count: int = 0       # How many times they've been helped

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    @property
    def has_complete_profile(self) -> bool:
        """Check if attendee has at least one skill OR one need."""
        return bool(self.superpowers) or bool(self.needs)


@dataclass
class Match:
    """
    Represents a pairing between two attendees for one round.

    For complementary matches (Rounds 1-3):
    - helper: The person GIVING help (sharing their skills)
    - helped: The person RECEIVING help (getting their needs met)

    For similarity matches (Round 4):
    - Both fields just hold the two people (no hierarchy)

    The reason and score come from the AI, explaining WHY this is a good match.
    """
    helper: Attendee            # Person A (or the helper in complementary rounds)
    helped: Attendee            # Person B (or the person being helped)
    reason: str                 # AI-generated explanation of the match
    score: int                  # 0-100 quality score from AI
    round_num: int              # Which round (1, 2, 3, or 4)
    match_type: str             # "complementary" or "similarity"
    third_person: Optional[Attendee] = None  # For trios when odd attendee count
    third_person_reason: Optional[str] = None  # Why the third person fits


# ============================================================================
# STEP 1: LOAD ATTENDEES FROM LUMA CSV
# ============================================================================
#
# Luma exports a CSV file with all registered attendees. This function reads
# that file and extracts the information we need for matching.
#
# We only include attendees who are "approved" or "checked_in" - this filters
# out cancelled registrations, waitlisted people, etc.
#
# ============================================================================

def parse_csv(filepath: str) -> list[Attendee]:
    """
    Read the Luma CSV export and create Attendee objects.

    The CSV has columns like:
    - name, email, approval_status
    - "What are your superpower (skills you have)?" → their skills
    - "Do you have any other superpowers we missed?" → additional skills
    - "What do you need help with right now?" → their needs
    - "Is there anything else you need?" → additional needs
    - "What project(s) are you working on?" → their project
    - "What phase are you in?" → Spark/Build/Momentum/Maintenance

    Returns a list of Attendee objects ready for matching.
    """
    attendees = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # ----------------------------------------------------------------
            # FILTER: Only include approved or checked-in attendees
            # This skips cancelled, declined, or waitlisted registrations
            # ----------------------------------------------------------------
            status = row.get('approval_status', '').lower()
            if status not in ['approved', 'checked_in']:
                continue

            # ----------------------------------------------------------------
            # EXTRACT SKILLS (Superpowers)
            # Combine the main superpowers field with the "anything else" field
            # ----------------------------------------------------------------
            superpowers_raw = row.get('What are your superpower (skills you have)🦸?', '')
            extra_skills = row.get('Do you have any other superpowers we missed?', '')

            # Split comma-separated skills into a list
            superpowers = [s.strip() for s in superpowers_raw.split(',') if s.strip()]

            # Add extra skills if they wrote something meaningful (not "no" or "n/a")
            if extra_skills and extra_skills.lower() not in ['no', 'n/a', 'na', 'none', '-']:
                superpowers.append(extra_skills.strip())

            # ----------------------------------------------------------------
            # EXTRACT NEEDS
            # Same pattern - combine main needs with "anything else" field
            # ----------------------------------------------------------------
            needs_raw = row.get('What do you need help with right now?🚁', '')
            extra_needs = row.get('Is there anything else you need?', '')

            needs = [n.strip() for n in needs_raw.split(',') if n.strip()]

            if extra_needs and extra_needs.lower() not in ['no', 'n/a', 'na', 'none', '-']:
                needs.append(extra_needs.strip())

            # ----------------------------------------------------------------
            # CREATE THE ATTENDEE OBJECT
            # ----------------------------------------------------------------
            attendee = Attendee(
                id=row.get('api_id', ''),
                name=row.get('name', '').strip(),
                email=row.get('email', ''),
                linkedin=row.get('LinkedIn', ''),
                project=row.get('What project(s) are you working on?', ''),
                phase=row.get('What phase are you in? ', '').strip(),
                superpowers=superpowers,
                needs=needs,
            )

            # Only add if they have a name (skip empty rows)
            if attendee.name:
                attendees.append(attendee)

    return attendees


# ============================================================================
# STEP 2A: AI-POWERED HELPER MATCHING (Rounds 1-3)
# ============================================================================
#
# This is where the magic happens! For each person who needs help, we ask
# the AI to look at everyone available and pick the BEST helper.
#
# The AI considers:
# - What skills does this person need?
# - Who has those skills?
# - How strong is the match?
#
# The AI returns:
# - The name of the best helper
# - A reason explaining why (shown to attendees!)
# - A score from 0-100
#
# ============================================================================

def find_best_helper_ai(
    person: Attendee,
    available_pool: list[Attendee],
    previous_matches: set[tuple[str, str]]
) -> tuple[Optional[Attendee], str, int]:
    """
    Use AI (GPT-4o-mini) to find the best helper for someone.

    Args:
        person: The attendee who needs help
        available_pool: List of people who could potentially help
        previous_matches: Set of (id, id) pairs who already met - can't match again!

    Returns:
        (helper, reason, score) or (None, "", 0) if no valid match found
    """
    if not available_pool:
        return None, "", 0

    # ----------------------------------------------------------------
    # FILTER: Remove people who already matched with this person
    # This ensures no one meets the same person twice!
    # ----------------------------------------------------------------
    valid_helpers = [
        h for h in available_pool
        if h.id != person.id  # Can't match with yourself
        and (person.id, h.id) not in previous_matches  # Haven't met before
        and (h.id, person.id) not in previous_matches  # (check both directions)
    ]

    if not valid_helpers:
        return None, "", 0

    # ----------------------------------------------------------------
    # BUILD THE AI PROMPT
    # We tell the AI about the person needing help and list all
    # available helpers with their superpowers
    # ----------------------------------------------------------------
    helpers_info = "\n".join([
        f"- {h.name}: Superpowers = {', '.join(h.superpowers) or 'None listed'}"
        for h in valid_helpers[:30]  # Limit to 30 to avoid hitting token limits
    ])

    prompt = f"""You are matching people at a networking event.

PERSON NEEDING HELP:
Name: {person.name}
Needs help with: {', '.join(person.needs) or 'No specific needs listed'}
Working on: {person.project}

AVAILABLE HELPERS:
{helpers_info}

Which ONE person from the available helpers can best help {person.name}?
Consider whose superpowers best match what {person.name} needs help with.

Respond in this exact JSON format only, no other text:
{{"best_helper_name": "Name of the best helper", "reason": "One sentence explaining why this is a good match", "score": 85}}

The score should be 0-100 based on how well the helper's skills match the person's needs.
If no one is a good match, still pick the best available option but give a lower score."""

    # ----------------------------------------------------------------
    # CALL THE AI
    # We use GPT-4o-mini for speed and cost efficiency
    # ----------------------------------------------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse the AI's response
        response_text = response.choices[0].message.content

        # Extract the JSON from the response
        import re
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())

            # Find the helper by name (the AI returns a name, we need the object)
            helper_name = result.get('best_helper_name', '')
            for h in valid_helpers:
                if h.name.lower() == helper_name.lower() or helper_name.lower() in h.name.lower():
                    return h, result.get('reason', ''), result.get('score', 50)

            # If exact name match not found, return the first valid helper
            return valid_helpers[0], result.get('reason', 'Best available match'), result.get('score', 50)

    except Exception as e:
        print(f"  AI call failed: {e}")

    # ----------------------------------------------------------------
    # FALLBACK: If AI fails, just return the first available person
    # Better to have a match than no match!
    # ----------------------------------------------------------------
    return valid_helpers[0] if valid_helpers else None, "Fallback match", 30


# ============================================================================
# STEP 2B: AI-POWERED SIMILARITY MATCHING (Round 4)
# ============================================================================
#
# Round 4 is different - instead of helper/helped, we match PEERS.
# We're looking for people working on similar things who can share
# experiences, struggles, and ideas as equals.
#
# The AI considers:
# - Similar projects (both building AI tools? both in e-commerce?)
# - Same phase (both in "Build" mode?)
# - Overlapping needs (both looking for fundraising help?)
# - Complementary perspectives (could learn from each other)
#
# ============================================================================

def find_similar_person_ai(
    person: Attendee,
    available_pool: list[Attendee],
    previous_matches: set[tuple[str, str]]
) -> tuple[Optional[Attendee], str, int]:
    """
    Use AI to find the most SIMILAR person for peer collaboration.

    Unlike helper matching, we're not looking for skill→need alignment.
    We want people on similar journeys who can relate and share.

    Args:
        person: The attendee to find a peer for
        available_pool: List of potential peers
        previous_matches: People who already met (can't match again)

    Returns:
        (peer, reason, score) or (None, "", 0) if no valid match
    """
    if not available_pool:
        return None, "", 0

    # Filter out people who already matched with this person
    valid_peers = [
        p for p in available_pool
        if p.id != person.id
        and (person.id, p.id) not in previous_matches
        and (p.id, person.id) not in previous_matches
    ]

    if not valid_peers:
        return None, "", 0

    # ----------------------------------------------------------------
    # BUILD THE AI PROMPT
    # For similarity, we share projects, phases, and needs
    # ----------------------------------------------------------------
    peers_info = "\n".join([
        f"- {p.name}: Project = {p.project}, Phase = {p.phase}, Needs = {', '.join(p.needs)}"
        for p in valid_peers[:30]
    ])

    prompt = f"""You are matching people at a networking event for a PEER COLLABORATION round.
The goal is to find someone working on SIMILAR things who can share ideas and experiences.

PERSON TO MATCH:
Name: {person.name}
Project: {person.project}
Phase: {person.phase}
Needs: {', '.join(person.needs)}
Skills: {', '.join(person.superpowers)}

AVAILABLE PEERS:
{peers_info}

Which ONE person is most SIMILAR to {person.name}?
Consider: similar project type, same phase, overlapping needs, complementary perspectives.

Respond in this exact JSON format only, no other text:
{{"best_peer_name": "Name of the most similar person", "reason": "One sentence explaining what they have in common and could discuss", "score": 85}}

Score 0-100 based on similarity."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.choices[0].message.content

        import re
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())

            peer_name = result.get('best_peer_name', '')
            for p in valid_peers:
                if p.name.lower() == peer_name.lower() or peer_name.lower() in p.name.lower():
                    return p, result.get('reason', ''), result.get('score', 50)

            return valid_peers[0], result.get('reason', 'Best available match'), result.get('score', 50)

    except Exception as e:
        print(f"  AI call failed: {e}")

    return valid_peers[0] if valid_peers else None, "Fallback match", 30


# ============================================================================
# STEP 2C: FIND BEST PAIR FOR LEFTOVER PERSON (Trio Formation)
# ============================================================================
#
# When there's an odd number of attendees, one person would be left out.
# Instead, we find the best existing pair for them to join, creating a trio.
#
# ============================================================================

def find_best_pair_for_leftover_ai(
    leftover: Attendee,
    matches: list[Match],
    match_type: str
) -> tuple[Optional[Match], str, int]:
    """
    Use AI to find the best existing pair for a leftover person to join.

    Args:
        leftover: The attendee who wasn't matched
        matches: List of pairs already created this round
        match_type: "complementary" or "similarity"

    Returns:
        (best_match, reason, score) or (None, "", 0) if no valid option
    """
    if not matches:
        return None, "", 0

    # Build info about available pairs
    pairs_info = []
    for i, match in enumerate(matches):
        if match_type == "complementary":
            pairs_info.append(
                f"Pair {i+1}: {match.helper.name} (helper, skills: {', '.join(match.helper.superpowers[:3])}) "
                f"↔ {match.helped.name} (helped, needs: {', '.join(match.helped.needs[:3])})"
            )
        else:
            pairs_info.append(
                f"Pair {i+1}: {match.helper.name} (project: {match.helper.project[:50]}) "
                f"↔ {match.helped.name} (project: {match.helped.project[:50]})"
            )

    pairs_text = "\n".join(pairs_info[:15])  # Limit to avoid token issues

    if match_type == "complementary":
        prompt = f"""You are matching people at a networking event.

LEFTOVER PERSON (needs to join a pair):
Name: {leftover.name}
Skills: {', '.join(leftover.superpowers) or 'None listed'}
Needs: {', '.join(leftover.needs) or 'None listed'}
Project: {leftover.project}

EXISTING PAIRS:
{pairs_text}

Which pair should {leftover.name} join to make a trio?
Consider: Can they help someone in the pair? Can someone in the pair help them?
A trio works best when all three can contribute something.

Respond in this exact JSON format only:
{{"best_pair_number": 1, "reason": "One sentence explaining how they fit with this pair", "score": 75}}

Score 0-100 based on how well they fit with the pair."""
    else:
        prompt = f"""You are matching people at a networking event for peer collaboration.

LEFTOVER PERSON (needs to join a pair):
Name: {leftover.name}
Project: {leftover.project}
Phase: {leftover.phase}
Skills: {', '.join(leftover.superpowers)}

EXISTING PAIRS:
{pairs_text}

Which pair should {leftover.name} join to make a trio?
Look for similar projects, phases, or complementary perspectives.

Respond in this exact JSON format only:
{{"best_pair_number": 1, "reason": "One sentence explaining what they all have in common", "score": 75}}

Score 0-100 based on similarity."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.choices[0].message.content

        import re
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            pair_num = result.get('best_pair_number', 1)
            # Convert to 0-indexed
            pair_idx = pair_num - 1
            if 0 <= pair_idx < len(matches):
                return matches[pair_idx], result.get('reason', ''), result.get('score', 50)

    except Exception as e:
        print(f"  AI call failed for trio matching: {e}")

    # Fallback: just pick the first pair
    return matches[0] if matches else None, "Fallback trio placement", 30


# ============================================================================
# STEP 2D: HEURISTIC SCORING FUNCTIONS (for optimal matching)
# ============================================================================
#
# These functions compute match scores WITHOUT calling the AI.
# They're fast enough to build a full NxN matrix of all possible pairs,
# which the Hungarian algorithm uses to find the globally optimal matching.
#
# AI is still used AFTER optimal pairs are found, just to generate
# human-readable explanations.
#
# ============================================================================

def compute_skill_need_overlap(helper: Attendee, helped: Attendee) -> float:
    """
    Compute how well helper's skills match helped's needs.
    Returns a score from 0-100.

    Uses keyword matching with fuzzy matching for related terms.
    """
    if not helper.superpowers or not helped.needs:
        return 20  # Base score if missing data

    # Normalize all strings for comparison
    helper_skills = [s.lower().strip() for s in helper.superpowers]
    helped_needs = [n.lower().strip() for n in helped.needs]

    # Common skill/need synonyms and related terms
    related_terms = {
        'marketing': ['growth', 'ads', 'advertising', 'social media', 'content', 'branding', 'seo'],
        'sales': ['revenue', 'deals', 'closing', 'business development', 'bd'],
        'engineering': ['coding', 'programming', 'development', 'software', 'tech', 'technical'],
        'design': ['ui', 'ux', 'user experience', 'visual', 'graphics', 'figma'],
        'product': ['pm', 'product management', 'roadmap', 'features'],
        'fundraising': ['funding', 'investors', 'vc', 'venture', 'capital', 'pitch'],
        'finance': ['accounting', 'financial', 'budgeting', 'money'],
        'legal': ['contracts', 'compliance', 'ip', 'intellectual property'],
        'hiring': ['recruiting', 'talent', 'team building', 'hr'],
        'ai': ['machine learning', 'ml', 'artificial intelligence', 'llm', 'gpt'],
        'data': ['analytics', 'metrics', 'analysis', 'insights'],
        'operations': ['ops', 'processes', 'systems', 'efficiency'],
        'strategy': ['planning', 'direction', 'vision', 'business model'],
    }

    def terms_match(skill: str, need: str) -> bool:
        """Check if a skill matches a need (exact or related)."""
        # Exact match
        if skill in need or need in skill:
            return True

        # Check related terms
        for category, terms in related_terms.items():
            skill_in_category = category in skill or any(t in skill for t in terms)
            need_in_category = category in need or any(t in need for t in terms)
            if skill_in_category and need_in_category:
                return True

        return False

    # Count matches
    matches = 0
    for skill in helper_skills:
        for need in helped_needs:
            if terms_match(skill, need):
                matches += 1
                break  # Count each skill once

    # Score based on match ratio
    if matches == 0:
        return 25  # No matches, but not zero (allow fallback)

    # More matches = higher score, with diminishing returns
    match_ratio = matches / max(len(helper_skills), len(helped_needs))
    score = 30 + (70 * min(match_ratio * 2, 1.0))  # Scale 30-100

    return score


def compute_similarity_score(person_a: Attendee, person_b: Attendee) -> float:
    """
    Compute how similar two people are (for Round 4 peer matching).
    Returns a score from 0-100.

    Considers: project similarity, phase match, overlapping needs/skills.
    """
    score = 0

    # Phase match (0-25 points)
    if person_a.phase and person_b.phase:
        if person_a.phase.lower() == person_b.phase.lower():
            score += 25
        elif _phases_adjacent(person_a.phase, person_b.phase):
            score += 15

    # Project keyword overlap (0-35 points)
    project_a_words = set(person_a.project.lower().split())
    project_b_words = set(person_b.project.lower().split())

    # Remove common words
    stop_words = {'a', 'an', 'the', 'and', 'or', 'for', 'to', 'of', 'in', 'on', 'is', 'are', 'i', 'my', 'am', 'we'}
    project_a_words -= stop_words
    project_b_words -= stop_words

    if project_a_words and project_b_words:
        overlap = len(project_a_words & project_b_words)
        total = len(project_a_words | project_b_words)
        if total > 0:
            score += 35 * (overlap / total)

    # Skill overlap (0-20 points) - they can learn from each other
    skills_a = set(s.lower() for s in person_a.superpowers)
    skills_b = set(s.lower() for s in person_b.superpowers)
    if skills_a and skills_b:
        overlap = len(skills_a & skills_b)
        if overlap > 0:
            score += min(20, overlap * 7)

    # Need overlap (0-20 points) - facing similar challenges
    needs_a = set(n.lower() for n in person_a.needs)
    needs_b = set(n.lower() for n in person_b.needs)
    if needs_a and needs_b:
        overlap = len(needs_a & needs_b)
        if overlap > 0:
            score += min(20, overlap * 7)

    return max(score, 20)  # Minimum 20 to allow fallback matching


def _phases_adjacent(phase_a: str, phase_b: str) -> bool:
    """Check if two phases are adjacent (e.g., Spark→Build)."""
    phase_order = ['spark', 'build', 'momentum', 'maintenance']
    try:
        idx_a = phase_order.index(phase_a.lower())
        idx_b = phase_order.index(phase_b.lower())
        return abs(idx_a - idx_b) == 1
    except ValueError:
        return False


# ============================================================================
# STEP 2E: OPTIMAL BIPARTITE MATCHING (Hungarian Algorithm)
# ============================================================================
#
# Given a list of attendees, find the optimal pairing that maximizes
# total match quality across everyone.
#
# ============================================================================

def find_optimal_pairs_complementary(
    attendees: list[Attendee],
    previous_matches: set[tuple[str, str]]
) -> list[tuple[Attendee, Attendee, float]]:
    """
    Find optimal helper→helped pairs using the Hungarian algorithm.

    Returns list of (helper, helped, score) tuples.
    """
    n = len(attendees)
    if n < 2:
        return []

    if not SCIPY_AVAILABLE:
        # Fall back to greedy if scipy not installed
        return _greedy_complementary_matching(attendees, previous_matches)

    # Build the cost matrix (we negate scores because Hungarian minimizes)
    # Rows = potential helpers, Cols = potential helped
    import numpy as np
    cost_matrix = np.zeros((n, n))

    for i, helper in enumerate(attendees):
        for j, helped in enumerate(attendees):
            if i == j:
                # Can't match with self
                cost_matrix[i][j] = 10000
            elif (helper.id, helped.id) in previous_matches or (helped.id, helper.id) in previous_matches:
                # Already met
                cost_matrix[i][j] = 10000
            else:
                # Compute score and negate (Hungarian minimizes)
                score = compute_skill_need_overlap(helper, helped)
                # Add role balance factor
                role_balance = (helper.helped_count - helper.helper_count) * 5
                adjusted_score = score + role_balance
                cost_matrix[i][j] = -adjusted_score

    # Run Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Extract valid pairs (where cost < 10000)
    pairs = []
    used = set()

    for row, col in zip(row_indices, col_indices):
        if row not in used and col not in used and cost_matrix[row][col] < 5000:
            helper = attendees[row]
            helped = attendees[col]
            score = -cost_matrix[row][col]
            pairs.append((helper, helped, score))
            used.add(row)
            used.add(col)

    return pairs


def find_optimal_pairs_similarity(
    attendees: list[Attendee],
    previous_matches: set[tuple[str, str]]
) -> list[tuple[Attendee, Attendee, float]]:
    """
    Find optimal peer pairs for similarity matching using Hungarian algorithm.

    Returns list of (person_a, person_b, score) tuples.
    """
    n = len(attendees)
    if n < 2:
        return []

    if not SCIPY_AVAILABLE:
        return _greedy_similarity_matching(attendees, previous_matches)

    import numpy as np

    # For similarity, the matrix is symmetric
    # We need to ensure each person appears in exactly one pair
    cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                cost_matrix[i][j] = 10000
            elif i > j:
                # Mirror the lower triangle
                cost_matrix[i][j] = cost_matrix[j][i]
            elif (attendees[i].id, attendees[j].id) in previous_matches or \
                 (attendees[j].id, attendees[i].id) in previous_matches:
                cost_matrix[i][j] = 10000
            else:
                score = compute_similarity_score(attendees[i], attendees[j])
                cost_matrix[i][j] = -score

    # Run Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Extract pairs (avoiding duplicates since matrix is symmetric)
    pairs = []
    used = set()

    for row, col in zip(row_indices, col_indices):
        if row < col and row not in used and col not in used and cost_matrix[row][col] < 5000:
            person_a = attendees[row]
            person_b = attendees[col]
            score = -cost_matrix[row][col]
            pairs.append((person_a, person_b, score))
            used.add(row)
            used.add(col)

    return pairs


def _greedy_complementary_matching(
    attendees: list[Attendee],
    previous_matches: set[tuple[str, str]]
) -> list[tuple[Attendee, Attendee, float]]:
    """Fallback greedy matching when scipy is not available."""
    pairs = []
    available = list(attendees)
    random.shuffle(available)

    while len(available) >= 2:
        helper = available.pop(0)
        best_helped = None
        best_score = -1

        for helped in available:
            if (helper.id, helped.id) in previous_matches or (helped.id, helper.id) in previous_matches:
                continue
            score = compute_skill_need_overlap(helper, helped)
            if score > best_score:
                best_score = score
                best_helped = helped

        if best_helped:
            pairs.append((helper, best_helped, best_score))
            available.remove(best_helped)

    return pairs


def _greedy_similarity_matching(
    attendees: list[Attendee],
    previous_matches: set[tuple[str, str]]
) -> list[tuple[Attendee, Attendee, float]]:
    """Fallback greedy similarity matching when scipy is not available."""
    pairs = []
    available = list(attendees)
    random.shuffle(available)

    while len(available) >= 2:
        person_a = available.pop(0)
        best_person = None
        best_score = -1

        for person_b in available:
            if (person_a.id, person_b.id) in previous_matches or (person_b.id, person_a.id) in previous_matches:
                continue
            score = compute_similarity_score(person_a, person_b)
            if score > best_score:
                best_score = score
                best_person = person_b

        if best_person:
            pairs.append((person_a, best_person, best_score))
            available.remove(best_person)

    return pairs


def generate_match_reason_ai(helper: Attendee, helped: Attendee, match_type: str) -> str:
    """
    Use AI to generate a human-readable explanation for why this match is good.
    Called AFTER optimal pairs are found.
    """
    if match_type == "complementary":
        prompt = f"""In one sentence, explain why {helper.name} (skills: {', '.join(helper.superpowers[:3])})
is a good match to help {helped.name} (needs: {', '.join(helped.needs[:3])}).
Be specific about which skill matches which need. Keep it under 20 words."""
    else:
        prompt = f"""In one sentence, explain what {helper.name} (project: {helper.project[:50]}, phase: {helper.phase})
and {helped.name} (project: {helped.project[:50]}, phase: {helped.phase}) have in common.
Keep it under 20 words."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if match_type == "complementary":
            return f"{helper.name} can help with {helped.needs[0] if helped.needs else 'their goals'}"
        else:
            return f"Both working on similar projects"


# ============================================================================
# STEP 3A: RUN A COMPLEMENTARY ROUND (Rounds 1, 2, 3)
# ============================================================================
#
# This function runs one round of "helper → helped" matching.
#
# THE ALGORITHM (with role balancing):
# 1. Sort attendees by who MOST NEEDS to be helped (low helped_count first)
# 2. For each person needing help, find a helper from those who:
#    - Have been helped more than they've helped (should give back)
#    - Or at least haven't been helper too many times
# 3. Create the match, remove BOTH people from the pool
# 4. Repeat until everyone is matched
#
# ROLE BALANCING STRATEGY:
# - Round 1: Everyone starts equal, just match by skills
# - Round 2: Prioritize R1 helpers to now be helped, and vice versa
# - Round 3: Force balance - people who helped 2x MUST be helped now
#
# ============================================================================

def run_complementary_round(
    round_num: int,
    attendees: list[Attendee],
    previous_matches: set[tuple[str, str]]
) -> tuple[list[Match], list[Attendee]]:
    """
    Run one round of complementary matching with GREEDY AI MATCHING.

    Args:
        round_num: Which round (1, 2, or 3)
        attendees: All attendees
        previous_matches: Who already met (updated as we go!)

    Returns:
        Tuple of (matches, unmatched_attendees)
    """
    print(f"\n{'='*60}")
    print(f"ROUND {round_num}: Complementary Matching")
    print(f"{'='*60}")

    matches = []
    available_pool = list(attendees)

    # Role balancing: prioritize people who need to be helped
    if round_num == 1:
        random.shuffle(available_pool)
    else:
        random.shuffle(available_pool)
        available_pool.sort(key=lambda p: (p.helper_count - p.helped_count), reverse=True)

    matched_this_round = set()

    iteration = 0
    while len(available_pool) >= 2:
        iteration += 1

        # Pick the next person who needs to be helped
        person = None
        for p in available_pool:
            if p.id not in matched_this_round:
                person = p
                break

        if not person:
            break

        print(f"\n[{iteration}] Finding helper for: {person.name}")

        # Find helpers, prioritizing those who should be helping
        pool_without_person = [p for p in available_pool if p.id != person.id and p.id not in matched_this_round]
        pool_without_person.sort(key=lambda p: (p.helped_count - p.helper_count), reverse=True)

        helper, reason, score = find_best_helper_ai(person, pool_without_person, previous_matches)

        if helper:
            print(f"    → {helper.name} (score: {score})")

            match = Match(
                helper=helper,
                helped=person,
                reason=reason,
                score=score,
                round_num=round_num,
                match_type="complementary"
            )
            matches.append(match)

            helper.helper_count += 1
            person.helped_count += 1
            matched_this_round.add(person.id)
            matched_this_round.add(helper.id)
            previous_matches.add((person.id, helper.id))

            available_pool = [p for p in available_pool if p.id not in matched_this_round]
        else:
            print(f"    → No valid helper found")
            matched_this_round.add(person.id)
            available_pool = [p for p in available_pool if p.id != person.id]

    # Find unmatched attendees
    unmatched = [a for a in attendees if a.id not in matched_this_round]

    # ----------------------------------------------------------------
    # TRIO FORMATION: If someone is left over, add them to a pair
    # ----------------------------------------------------------------
    if len(unmatched) == 1 and matches:
        leftover = unmatched[0]
        print(f"\n[TRIO] Finding best pair for leftover: {leftover.name}")

        best_match, trio_reason, trio_score = find_best_pair_for_leftover_ai(
            leftover, matches, "complementary"
        )

        if best_match:
            # Add leftover as third person to this match
            best_match.third_person = leftover
            best_match.third_person_reason = trio_reason

            # Update tracking - leftover gets credit for being in a match
            # In a trio, they can both help and be helped
            leftover.helper_count += 0.5
            leftover.helped_count += 0.5
            matched_this_round.add(leftover.id)
            # Record that leftover has now "met" both people in the pair
            previous_matches.add((leftover.id, best_match.helper.id))
            previous_matches.add((leftover.id, best_match.helped.id))

            print(f"    → Joined trio with: {best_match.helper.name} & {best_match.helped.name}")
            print(f"    → Reason: {trio_reason}")

            unmatched = []  # No longer anyone unmatched

    print(f"\nRound {round_num} complete: {len(matches)} matches created")
    trio_count = len([m for m in matches if m.third_person])
    if trio_count:
        print(f"  (includes {trio_count} trio)")
    if unmatched:
        print(f"  ⚠ Unmatched this round: {', '.join([u.name for u in unmatched])}")

    return matches, unmatched


# ============================================================================
# STEP 3B: RUN THE SIMILARITY ROUND (Round 4)
# ============================================================================
#
# Same structure as complementary rounds, but:
# - Uses find_similar_person_ai instead of find_best_helper_ai
# - No helper/helped distinction - both are equals
# - Optimizes for similarity instead of skill→need matching
#
# ============================================================================

def run_similarity_round(
    round_num: int,
    attendees: list[Attendee],
    previous_matches: set[tuple[str, str]]
) -> tuple[list[Match], list[Attendee]]:
    """
    Run the peer collaboration round (Round 4) with GREEDY AI MATCHING.

    Returns:
        Tuple of (matches, unmatched_attendees)
    """
    print(f"\n{'='*60}")
    print(f"ROUND {round_num}: Similarity Matching (Peer Collab)")
    print(f"{'='*60}")

    matches = []
    available_pool = list(attendees)
    random.shuffle(available_pool)

    matched_this_round = set()

    iteration = 0
    while len(available_pool) >= 2:
        iteration += 1

        person = None
        for p in available_pool:
            if p.id not in matched_this_round:
                person = p
                break

        if not person:
            break

        print(f"\n[{iteration}] Finding peer for: {person.name}")

        pool_without_person = [p for p in available_pool if p.id != person.id and p.id not in matched_this_round]

        peer, reason, score = find_similar_person_ai(person, pool_without_person, previous_matches)

        if peer:
            print(f"    → {peer.name} (score: {score})")

            match = Match(
                helper=person,
                helped=peer,
                reason=reason,
                score=score,
                round_num=round_num,
                match_type="similarity"
            )
            matches.append(match)

            matched_this_round.add(person.id)
            matched_this_round.add(peer.id)
            previous_matches.add((person.id, peer.id))

            available_pool = [p for p in available_pool if p.id not in matched_this_round]
        else:
            print(f"    → No valid peer found")
            matched_this_round.add(person.id)
            available_pool = [p for p in available_pool if p.id != person.id]

    # Find unmatched attendees
    unmatched = [a for a in attendees if a.id not in matched_this_round]

    # ----------------------------------------------------------------
    # TRIO FORMATION: If someone is left over, add them to a pair
    # ----------------------------------------------------------------
    if len(unmatched) == 1 and matches:
        leftover = unmatched[0]
        print(f"\n[TRIO] Finding best pair for leftover: {leftover.name}")

        best_match, trio_reason, trio_score = find_best_pair_for_leftover_ai(
            leftover, matches, "similarity"
        )

        if best_match:
            # Add leftover as third person to this match
            best_match.third_person = leftover
            best_match.third_person_reason = trio_reason

            matched_this_round.add(leftover.id)
            # Record that leftover has now "met" both people in the pair
            previous_matches.add((leftover.id, best_match.helper.id))
            previous_matches.add((leftover.id, best_match.helped.id))

            print(f"    → Joined trio with: {best_match.helper.name} & {best_match.helped.name}")
            print(f"    → Reason: {trio_reason}")

            unmatched = []  # No longer anyone unmatched

    print(f"\nRound {round_num} complete: {len(matches)} matches created")
    trio_count = len([m for m in matches if m.third_person])
    if trio_count:
        print(f"  (includes {trio_count} trio)")
    if unmatched:
        print(f"  ⚠ Unmatched this round: {', '.join([u.name for u in unmatched])}")

    return matches, unmatched


# ============================================================================
# STEP 4: RUN ALL 4 ROUNDS
# ============================================================================
#
# This orchestrates the entire matching process:
# 1. Run Round 1 (complementary)
# 2. Run Round 2 (complementary)
# 3. Run Round 3 (complementary)
# 4. Run Round 4 (similarity)
#
# The previous_matches set carries across all rounds, ensuring no repeats.
#
# ============================================================================

def run_all_rounds(attendees: list[Attendee]) -> dict:
    """
    Run all 4 rounds of matching and compile results.

    Returns a dictionary ready to be saved as JSON with:
    - Total attendee count
    - Total match count
    - Per-round match details
    """
    all_matches = []

    # This set tracks ALL pairings across all rounds
    # Format: {(person1_id, person2_id), ...}
    previous_matches: set[tuple[str, str]] = set()

    # ----------------------------------------------------------------
    # ROUNDS 1-3: Complementary matching (skill → need)
    # ----------------------------------------------------------------
    for round_num in range(1, 4):
        round_matches, _ = run_complementary_round(round_num, attendees, previous_matches)
        all_matches.extend(round_matches)

    # ----------------------------------------------------------------
    # ROUND 4: Similarity matching (peer collaboration)
    # ----------------------------------------------------------------
    round_4_matches, _ = run_similarity_round(4, attendees, previous_matches)
    all_matches.extend(round_4_matches)

    # ----------------------------------------------------------------
    # BUILD THE OUTPUT JSON STRUCTURE
    # ----------------------------------------------------------------
    output = {
        "total_attendees": len(attendees),
        "total_matches": len(all_matches),
        "rounds": []
    }

    for round_num in range(1, 5):
        round_matches = [m for m in all_matches if m.round_num == round_num]
        round_type = "similarity" if round_num == 4 else "complementary"

        round_data = {
            "round": round_num,
            "type": round_type,
            "match_count": len(round_matches),
            "matches": []
        }

        for match in round_matches:
            if match.match_type == "complementary":
                # For helper→helped rounds, show who's helping whom
                match_data = {
                    "helper": {
                        "name": match.helper.name,
                        "linkedin": match.helper.linkedin,
                        "superpowers": match.helper.superpowers
                    },
                    "helped": {
                        "name": match.helped.name,
                        "linkedin": match.helped.linkedin,
                        "needs": match.helped.needs
                    },
                    "reason": match.reason,
                    "score": match.score,
                    "is_trio": match.third_person is not None
                }
                # Add third person if this is a trio
                if match.third_person:
                    match_data["third_person"] = {
                        "name": match.third_person.name,
                        "linkedin": match.third_person.linkedin,
                        "superpowers": match.third_person.superpowers,
                        "needs": match.third_person.needs
                    }
                    match_data["third_person_reason"] = match.third_person_reason
            else:
                # For peer rounds, show both as equals
                match_data = {
                    "person_a": {
                        "name": match.helper.name,
                        "linkedin": match.helper.linkedin,
                        "project": match.helper.project
                    },
                    "person_b": {
                        "name": match.helped.name,
                        "linkedin": match.helped.linkedin,
                        "project": match.helped.project
                    },
                    "similarity_reason": match.reason,
                    "score": match.score,
                    "is_trio": match.third_person is not None
                }
                # Add third person if this is a trio
                if match.third_person:
                    match_data["person_c"] = {
                        "name": match.third_person.name,
                        "linkedin": match.third_person.linkedin,
                        "project": match.third_person.project
                    }
                    match_data["third_person_reason"] = match.third_person_reason

            round_data["matches"].append(match_data)

        output["rounds"].append(round_data)

    return output


# ============================================================================
# STEP 5: EXPORT MATCHES TO CSV
# ============================================================================
#
# Creates easy-to-read CSV files for printing or sharing:
# - One master CSV with all rounds
# - Separate CSV per round (for printing table tents or handouts)
#
# ============================================================================

def export_to_csv(results: dict, output_dir: str):
    """
    Export matches to CSV files for easy printing/sharing.

    Creates:
    - matches_all.csv - All rounds in one file
    - round1.csv, round2.csv, etc. - One file per round
    """

    # ----------------------------------------------------------------
    # MASTER CSV: All rounds in one file
    # ----------------------------------------------------------------
    all_matches_path = os.path.join(output_dir, 'matches_all.csv')

    with open(all_matches_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header row (with trio support)
        writer.writerow([
            'Round',
            'Type',
            'Person A (Helper/Peer)',
            'Person A LinkedIn',
            'Person B (Helped/Peer)',
            'Person B LinkedIn',
            'Person C (Trio)',
            'Person C LinkedIn',
            'Match Reason',
            'Trio Reason',
            'Score'
        ])

        # Write each match
        for round_data in results['rounds']:
            round_num = round_data['round']
            round_type = round_data['type']

            for match in round_data['matches']:
                is_trio = match.get('is_trio', False)

                if round_type == 'complementary':
                    third_name = match.get('third_person', {}).get('name', '') if is_trio else ''
                    third_linkedin = match.get('third_person', {}).get('linkedin', '') if is_trio else ''
                    trio_reason = match.get('third_person_reason', '') if is_trio else ''

                    writer.writerow([
                        round_num,
                        'Trio' if is_trio else 'Helper → Helped',
                        match['helper']['name'],
                        match['helper']['linkedin'],
                        match['helped']['name'],
                        match['helped']['linkedin'],
                        third_name,
                        third_linkedin,
                        match['reason'],
                        trio_reason,
                        match['score']
                    ])
                else:
                    third_name = match.get('person_c', {}).get('name', '') if is_trio else ''
                    third_linkedin = match.get('person_c', {}).get('linkedin', '') if is_trio else ''
                    trio_reason = match.get('third_person_reason', '') if is_trio else ''

                    writer.writerow([
                        round_num,
                        'Trio' if is_trio else 'Peer Collab',
                        match['person_a']['name'],
                        match['person_a']['linkedin'],
                        match['person_b']['name'],
                        match['person_b']['linkedin'],
                        third_name,
                        third_linkedin,
                        match['similarity_reason'],
                        trio_reason,
                        match['score']
                    ])

    print(f"  → Saved: {all_matches_path}")

    # ----------------------------------------------------------------
    # PER-ROUND CSVs: One file per round (great for printing)
    # ----------------------------------------------------------------
    for round_data in results['rounds']:
        round_num = round_data['round']
        round_type = round_data['type']
        round_path = os.path.join(output_dir, f'round{round_num}.csv')

        with open(round_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            if round_type == 'complementary':
                # Header for helper/helped rounds (with trio support)
                writer.writerow([
                    'Match #',
                    'Type',
                    'Helper Name',
                    'Helper LinkedIn',
                    'Helped Name',
                    'Helped LinkedIn',
                    'Third Person',
                    'Third LinkedIn',
                    'Why This Match?',
                    'Trio Reason',
                    'Score'
                ])

                for i, match in enumerate(round_data['matches'], 1):
                    is_trio = match.get('is_trio', False)
                    third_name = match.get('third_person', {}).get('name', '') if is_trio else ''
                    third_linkedin = match.get('third_person', {}).get('linkedin', '') if is_trio else ''
                    trio_reason = match.get('third_person_reason', '') if is_trio else ''

                    writer.writerow([
                        i,
                        'TRIO' if is_trio else 'Pair',
                        match['helper']['name'],
                        match['helper']['linkedin'],
                        match['helped']['name'],
                        match['helped']['linkedin'],
                        third_name,
                        third_linkedin,
                        match['reason'],
                        trio_reason,
                        match['score']
                    ])
            else:
                # Header for peer/similarity rounds (with trio support)
                writer.writerow([
                    'Match #',
                    'Type',
                    'Person A',
                    'Person A LinkedIn',
                    'Person A Project',
                    'Person B',
                    'Person B LinkedIn',
                    'Person B Project',
                    'Person C',
                    'Person C LinkedIn',
                    'Person C Project',
                    'What They Have In Common',
                    'Trio Reason',
                    'Score'
                ])

                for i, match in enumerate(round_data['matches'], 1):
                    is_trio = match.get('is_trio', False)
                    third_name = match.get('person_c', {}).get('name', '') if is_trio else ''
                    third_linkedin = match.get('person_c', {}).get('linkedin', '') if is_trio else ''
                    third_project = match.get('person_c', {}).get('project', '') if is_trio else ''
                    trio_reason = match.get('third_person_reason', '') if is_trio else ''

                    writer.writerow([
                        i,
                        'TRIO' if is_trio else 'Pair',
                        match['person_a']['name'],
                        match['person_a']['linkedin'],
                        match['person_a']['project'],
                        match['person_b']['name'],
                        match['person_b']['linkedin'],
                        match['person_b']['project'],
                        third_name,
                        third_linkedin,
                        third_project,
                        match['similarity_reason'],
                        trio_reason,
                        match['score']
                    ])

        print(f"  → Saved: {round_path}")


# ============================================================================
# INTERACTIVE STEP-BY-STEP HELPERS
# ============================================================================

def wait_for_user(step_name: str):
    """Pause and wait for user to review before continuing."""
    print(f"\n{'─'*60}")
    print(f"✓ {step_name} complete")
    print(f"{'─'*60}")
    input("Press Enter to continue to next step...")
    print()


def display_attendees_summary(attendees: list[Attendee]):
    """Show a summary of loaded attendees for review."""
    print(f"\n{'='*60}")
    print(f"LOADED ATTENDEES ({len(attendees)} total)")
    print(f"{'='*60}\n")

    # Group by phase
    phases = {}
    for a in attendees:
        phase = a.phase or "Unknown"
        if phase not in phases:
            phases[phase] = []
        phases[phase].append(a)

    for phase, people in sorted(phases.items()):
        print(f"\n{phase} ({len(people)} people):")
        for p in people:
            skills = ', '.join(p.superpowers[:3]) if p.superpowers else 'None'
            if len(p.superpowers) > 3:
                skills += f" (+{len(p.superpowers)-3} more)"
            needs = ', '.join(p.needs[:2]) if p.needs else 'None'
            if len(p.needs) > 2:
                needs += f" (+{len(p.needs)-2} more)"
            print(f"  • {p.name}")
            print(f"    Project: {p.project[:50]}{'...' if len(p.project) > 50 else ''}")
            print(f"    Skills: {skills}")
            print(f"    Needs: {needs}")


def display_round_matches(round_data: dict):
    """Show matches from a single round in a readable format."""
    round_num = round_data['round']
    round_type = round_data['type']
    matches = round_data['matches']
    unmatched = round_data.get('unmatched', [])
    trio_count = round_data.get('trio_count', 0)

    print(f"\n{'='*60}")
    header = f"ROUND {round_num} RESULTS: {round_type.upper()} ({len(matches)} matches"
    if trio_count:
        header += f", {trio_count} trio"
    header += ")"
    print(header)
    print(f"{'='*60}\n")

    for i, match in enumerate(matches, 1):
        is_trio = match.get('is_trio', False)

        if round_type == 'complementary':
            helper = match['helper']['name']
            helped = match['helped']['name']
            reason = match['reason']
            score = match['score']

            if is_trio:
                third = match['third_person']['name']
                trio_reason = match.get('third_person_reason', '')
                print(f"Match {i} [TRIO]: {helper} → {helped} + {third}")
                print(f"  Score: {score}/100")
                print(f"  Why: {reason}")
                print(f"  Trio fit: {trio_reason}")
            else:
                print(f"Match {i}: {helper} → {helped}")
                print(f"  Score: {score}/100")
                print(f"  Why: {reason}")
            print()
        else:
            person_a = match['person_a']['name']
            person_b = match['person_b']['name']
            reason = match['similarity_reason']
            score = match['score']

            if is_trio:
                person_c = match['person_c']['name']
                trio_reason = match.get('third_person_reason', '')
                print(f"Match {i} [TRIO]: {person_a} ↔ {person_b} ↔ {person_c}")
                print(f"  Score: {score}/100")
                print(f"  Common ground: {reason}")
                print(f"  Trio fit: {trio_reason}")
            else:
                print(f"Match {i}: {person_a} ↔ {person_b}")
                print(f"  Score: {score}/100")
                print(f"  Common ground: {reason}")
            print()

    # Show unmatched attendees
    if unmatched:
        print(f"{'─'*60}")
        print(f"SITTING OUT THIS ROUND ({len(unmatched)}):")
        for u in unmatched:
            print(f"  - {u['name']} - {u['project'][:40]}{'...' if len(u['project']) > 40 else ''}")
        print()


def run_all_rounds_interactive(attendees: list[Attendee], csv_path: str) -> dict:
    """
    Run all 4 rounds of matching with pauses between each step.
    Allows user to review output before continuing.
    """
    all_matches = []
    previous_matches: set[tuple[str, str]] = set()

    output = {
        "total_attendees": len(attendees),
        "total_matches": 0,
        "rounds": []
    }

    # ----------------------------------------------------------------
    # ROUNDS 1-3: Complementary matching (skill → need)
    # ----------------------------------------------------------------
    for round_num in range(1, 4):
        round_matches, unmatched = run_complementary_round(round_num, attendees, previous_matches)
        all_matches.extend(round_matches)

        # Build round data for display
        round_data = build_round_data(round_num, round_matches, "complementary", unmatched)
        output["rounds"].append(round_data)

        # Show results and pause
        display_round_matches(round_data)
        wait_for_user(f"Round {round_num}")

    # ----------------------------------------------------------------
    # ROUND 4: Similarity matching (peer collaboration)
    # ----------------------------------------------------------------
    round_4_matches, unmatched = run_similarity_round(4, attendees, previous_matches)
    all_matches.extend(round_4_matches)

    round_data = build_round_data(4, round_4_matches, "similarity", unmatched)
    output["rounds"].append(round_data)

    display_round_matches(round_data)
    wait_for_user("Round 4")

    output["total_matches"] = len(all_matches)
    return output


def build_round_data(round_num: int, matches: list[Match], round_type: str, unmatched: list[Attendee] = None) -> dict:
    """Convert Match objects to JSON-serializable dict for a single round."""
    round_data = {
        "round": round_num,
        "type": round_type,
        "match_count": len(matches),
        "trio_count": len([m for m in matches if m.third_person]),
        "unmatched": [{"name": u.name, "project": u.project} for u in (unmatched or [])],
        "matches": []
    }

    for match in matches:
        if match.match_type == "complementary":
            match_data = {
                "helper": {
                    "name": match.helper.name,
                    "linkedin": match.helper.linkedin,
                    "superpowers": match.helper.superpowers
                },
                "helped": {
                    "name": match.helped.name,
                    "linkedin": match.helped.linkedin,
                    "needs": match.helped.needs
                },
                "reason": match.reason,
                "score": match.score,
                "is_trio": match.third_person is not None
            }
            if match.third_person:
                match_data["third_person"] = {
                    "name": match.third_person.name,
                    "linkedin": match.third_person.linkedin,
                    "superpowers": match.third_person.superpowers,
                    "needs": match.third_person.needs
                }
                match_data["third_person_reason"] = match.third_person_reason
        else:
            match_data = {
                "person_a": {
                    "name": match.helper.name,
                    "linkedin": match.helper.linkedin,
                    "project": match.helper.project
                },
                "person_b": {
                    "name": match.helped.name,
                    "linkedin": match.helped.linkedin,
                    "project": match.helped.project
                },
                "similarity_reason": match.reason,
                "score": match.score,
                "is_trio": match.third_person is not None
            }
            if match.third_person:
                match_data["person_c"] = {
                    "name": match.third_person.name,
                    "linkedin": match.third_person.linkedin,
                    "project": match.third_person.project
                }
                match_data["third_person_reason"] = match.third_person_reason
        round_data["matches"].append(match_data)

    return round_data


# ============================================================================
# PRE-FLIGHT CHECK - Validate before matching starts
# ============================================================================
#
# Before running the matching algorithm, we check:
# - Do we have enough attendees for 4 rounds of unique matches?
# - Will there be trios (odd number)?
# - Are there incomplete profiles that will have poor matches?
# - Any other issues that could prevent successful matching?
#
# ============================================================================

def run_preflight_check(attendees: list[Attendee]) -> dict:
    """
    Validate that matching can succeed before running.

    Returns a dict with:
    - can_proceed: bool - whether matching can run
    - warnings: list of warning messages
    - errors: list of blocking errors
    - stats: useful statistics about the attendee pool
    """
    errors = []
    warnings = []
    stats = {
        "total_attendees": len(attendees),
        "has_odd_count": len(attendees) % 2 == 1,
        "incomplete_profiles": 0,
        "no_skills": 0,
        "no_needs": 0,
        "max_possible_unique_matches": len(attendees) - 1,
        "rounds_requested": 4
    }

    # ----------------------------------------------------------------
    # CHECK 1: Minimum attendee count
    # Need at least 5 people for 4 rounds of unique matches
    # (each person needs 4 different partners)
    # ----------------------------------------------------------------
    if len(attendees) < 2:
        errors.append("Need at least 2 attendees to run matching")
    elif len(attendees) < 5:
        errors.append(
            f"Only {len(attendees)} attendees - need at least 5 for 4 rounds of unique matches. "
            f"Each person can only meet {len(attendees) - 1} others."
        )

    # ----------------------------------------------------------------
    # CHECK 2: Odd number warning (will create trios)
    # ----------------------------------------------------------------
    if len(attendees) % 2 == 1:
        warnings.append(
            f"Odd number of attendees ({len(attendees)}). "
            f"Each round will have one trio (3-person group)."
        )

    # ----------------------------------------------------------------
    # CHECK 3: Profile completeness
    # ----------------------------------------------------------------
    incomplete = []
    no_skills = []
    no_needs = []

    for a in attendees:
        has_skills = bool(a.superpowers)
        has_needs = bool(a.needs)

        if not has_skills:
            no_skills.append(a.name)
        if not has_needs:
            no_needs.append(a.name)
        if not has_skills and not has_needs:
            incomplete.append(a.name)

    stats["incomplete_profiles"] = len(incomplete)
    stats["no_skills"] = len(no_skills)
    stats["no_needs"] = len(no_needs)

    if incomplete:
        warnings.append(
            f"{len(incomplete)} attendee(s) have NO skills AND NO needs listed - "
            f"matching quality will be poor: {', '.join(incomplete[:5])}"
            + (f" (+{len(incomplete)-5} more)" if len(incomplete) > 5 else "")
        )

    if len(no_skills) > len(attendees) * 0.3:
        warnings.append(
            f"{len(no_skills)} attendee(s) ({len(no_skills)*100//len(attendees)}%) have no skills listed. "
            f"Complementary matching (Rounds 1-3) may struggle to find helpers."
        )

    if len(no_needs) > len(attendees) * 0.3:
        warnings.append(
            f"{len(no_needs)} attendee(s) ({len(no_needs)*100//len(attendees)}%) have no needs listed. "
            f"Complementary matching may struggle to find people to help."
        )

    # ----------------------------------------------------------------
    # CHECK 4: Theoretical match capacity
    # With N people and 4 rounds, can everyone get 4 unique matches?
    # ----------------------------------------------------------------
    if len(attendees) >= 5:
        # With trios, someone in a trio meets 2 new people
        # Worst case: same person is in trio every round
        # They'd meet: 2 + 2 + 2 + 2 = 8 people (fine)
        # But they need to not repeat partners

        # The real constraint: with N attendees, each person
        # can meet at most N-1 unique people
        max_unique = len(attendees) - 1
        if max_unique < 4:
            # This is already caught in CHECK 1, but let's be explicit
            pass
        elif max_unique == 4:
            warnings.append(
                f"Exactly {len(attendees)} attendees = tight fit. "
                f"Each person will meet everyone else exactly once. "
                f"No room for optimization - every match is forced."
            )
        elif max_unique < 8:
            # With trios, some people meet 2 per round
            # If they're in trios every round: 2*4 = 8 meetings
            # But we only have max_unique people
            warnings.append(
                f"With {len(attendees)} attendees and potential trios, "
                f"some people may meet the same person twice across rounds if "
                f"they're in multiple trios."
            )

    # ----------------------------------------------------------------
    # CHECK 5: Skill/need diversity
    # If everyone has the same skill or same need, matching is hard
    # ----------------------------------------------------------------
    all_skills = []
    all_needs = []
    for a in attendees:
        all_skills.extend(a.superpowers)
        all_needs.extend(a.needs)

    unique_skills = len(set(all_skills))
    unique_needs = len(set(all_needs))

    stats["unique_skills"] = unique_skills
    stats["unique_needs"] = unique_needs

    if unique_skills < 3 and len(attendees) > 4:
        warnings.append(
            f"Low skill diversity: only {unique_skills} unique skills across all attendees. "
            f"Matching variety may be limited."
        )

    if unique_needs < 3 and len(attendees) > 4:
        warnings.append(
            f"Low need diversity: only {unique_needs} unique needs across all attendees. "
            f"Matching variety may be limited."
        )

    # ----------------------------------------------------------------
    # FINAL DETERMINATION
    # ----------------------------------------------------------------
    can_proceed = len(errors) == 0

    return {
        "can_proceed": can_proceed,
        "errors": errors,
        "warnings": warnings,
        "stats": stats
    }


def print_preflight_report(report: dict) -> bool:
    """
    Print the pre-flight check results.
    Returns True if matching can proceed, False if blocked.
    """
    print(f"\n{'='*60}")
    print("PRE-FLIGHT CHECK")
    print(f"{'='*60}")

    stats = report["stats"]
    print(f"\nAttendees: {stats['total_attendees']}")
    print(f"Max unique matches possible per person: {stats['max_possible_unique_matches']}")
    print(f"Rounds requested: {stats['rounds_requested']}")

    if stats["has_odd_count"]:
        print(f"Odd count: Yes (will create trios)")

    if stats.get("unique_skills"):
        print(f"Skill diversity: {stats['unique_skills']} unique skills")
    if stats.get("unique_needs"):
        print(f"Need diversity: {stats['unique_needs']} unique needs")

    # Show errors (blocking)
    if report["errors"]:
        print(f"\n{'─'*60}")
        print("ERRORS (blocking):")
        print(f"{'─'*60}")
        for error in report["errors"]:
            print(f"  [X] {error}")

    # Show warnings (non-blocking)
    if report["warnings"]:
        print(f"\n{'─'*60}")
        print("WARNINGS:")
        print(f"{'─'*60}")
        for warning in report["warnings"]:
            print(f"  [!] {warning}")

    # Final status
    print(f"\n{'='*60}")
    if report["can_proceed"]:
        if report["warnings"]:
            print("STATUS: Ready to proceed (with warnings)")
        else:
            print("STATUS: All checks passed!")
    else:
        print("STATUS: Cannot proceed - fix errors above")
    print(f"{'='*60}\n")

    return report["can_proceed"]


# ============================================================================
# AI VALIDATION - Review match quality with AI
# ============================================================================
#
# After matching is complete, we use AI to review the matches and flag
# any issues that the programmatic checks might miss:
# - Poor quality matches (skills don't actually align)
# - Role imbalances (someone always helping, never helped)
# - Missed opportunities
#
# ============================================================================

def run_ai_validation(results: dict, attendees: list[Attendee]) -> dict:
    """
    Use AI to review match quality and flag issues.

    Returns a dict with:
    - issues: list of problems found
    - suggestions: list of improvements
    - overall_quality: AI's assessment of overall match quality
    """
    print(f"\n  Reviewing matches with AI...")

    # Build a summary of all matches for AI review
    match_summary = []

    for round_data in results["rounds"]:
        round_num = round_data["round"]
        round_type = round_data["type"]

        for match in round_data["matches"]:
            if round_type == "complementary":
                match_summary.append({
                    "round": round_num,
                    "type": "helper→helped",
                    "helper": match["helper"]["name"],
                    "helper_skills": match["helper"].get("superpowers", [])[:3],
                    "helped": match["helped"]["name"],
                    "helped_needs": match["helped"].get("needs", [])[:3],
                    "score": match["score"],
                    "reason": match["reason"]
                })
            else:
                match_summary.append({
                    "round": round_num,
                    "type": "peer",
                    "person_a": match["person_a"]["name"],
                    "person_b": match["person_b"]["name"],
                    "score": match["score"],
                    "reason": match.get("similarity_reason", "")
                })

    # Build role balance summary
    role_summary = []
    for a in attendees:
        if abs(a.helper_count - a.helped_count) >= 2:
            role_summary.append(f"{a.name}: helped others {int(a.helper_count)}x, was helped {int(a.helped_count)}x")

    # Find low-score matches
    low_score_matches = [m for m in match_summary if m["score"] < 70]

    # Build the prompt
    prompt = f"""You are reviewing matches from a speed networking event.
Analyze these matches and identify any issues.

TOTAL MATCHES: {len(match_summary)}
LOW SCORE MATCHES (<70): {len(low_score_matches)}

{"LOW SCORE MATCHES:" if low_score_matches else ""}
{chr(10).join([f"- Round {m['round']}: {m.get('helper', m.get('person_a'))} → {m.get('helped', m.get('person_b'))} (score: {m['score']}) - {m.get('reason', m.get('similarity_reason', ''))}" for m in low_score_matches[:10]])}

{"ROLE IMBALANCES:" if role_summary else "No role imbalances."}
{chr(10).join(role_summary[:10]) if role_summary else ""}

Please provide:
1. ISSUES: List 0-5 specific problems (if any)
2. SUGGESTIONS: List 0-3 actionable improvements for next time
3. OVERALL: One sentence assessment of match quality (Good/Acceptable/Needs Improvement)

Be concise. If matches look good, say so. Format as:

ISSUES:
- issue 1
- issue 2

SUGGESTIONS:
- suggestion 1

OVERALL: Your assessment here"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        ai_response = response.choices[0].message.content.strip()

        # Parse the response
        issues = []
        suggestions = []
        overall = ""

        current_section = None
        for line in ai_response.split('\n'):
            line = line.strip()
            if line.startswith("ISSUES:"):
                current_section = "issues"
            elif line.startswith("SUGGESTIONS:"):
                current_section = "suggestions"
            elif line.startswith("OVERALL:"):
                current_section = "overall"
                overall = line.replace("OVERALL:", "").strip()
            elif line.startswith("- ") or line.startswith("• "):
                item = line[2:].strip()
                if current_section == "issues" and item:
                    issues.append(item)
                elif current_section == "suggestions" and item:
                    suggestions.append(item)
            elif current_section == "overall" and line and not overall:
                overall = line

        return {
            "issues": issues,
            "suggestions": suggestions,
            "overall": overall or "Assessment not available",
            "low_score_count": len(low_score_matches),
            "role_imbalances": role_summary,
            "raw_response": ai_response
        }

    except Exception as e:
        print(f"  AI validation failed: {e}")
        return {
            "issues": [],
            "suggestions": [],
            "overall": f"AI validation error: {e}",
            "low_score_count": len(low_score_matches),
            "role_imbalances": role_summary,
            "raw_response": ""
        }


def print_ai_validation(validation: dict):
    """Print the AI validation results."""
    print(f"\n{'='*60}")
    print("AI VALIDATION")
    print(f"{'='*60}")

    # Overall assessment
    print(f"\n{validation['overall']}")

    # Issues
    if validation["issues"]:
        print(f"\n{'─'*60}")
        print("ISSUES:")
        print(f"{'─'*60}")
        for issue in validation["issues"]:
            print(f"  [!] {issue}")
    else:
        print(f"\n  No issues found.")

    # Suggestions
    if validation["suggestions"]:
        print(f"\n{'─'*60}")
        print("SUGGESTIONS FOR NEXT TIME:")
        print(f"{'─'*60}")
        for suggestion in validation["suggestions"]:
            print(f"  → {suggestion}")

    # Stats
    print(f"\n{'─'*60}")
    print("STATS:")
    print(f"{'─'*60}")
    print(f"  Low score matches (<70): {validation['low_score_count']}")
    print(f"  Role imbalances: {len(validation['role_imbalances'])}")

    print(f"\n{'='*60}\n")


# ============================================================================
# COVERAGE REPORT - Verify everyone got matched
# ============================================================================
#
# This is the validation layer. After all rounds complete, we check:
# - Did everyone get 4 matches? (one per round)
# - Who got left out and when?
# - What's the match quality distribution?
#
# ============================================================================

def generate_coverage_report(results: dict, attendees: list[Attendee]) -> dict:
    """
    Analyze match results to verify everyone was matched properly.

    Returns a report dict with:
    - per_person: match count and details for each attendee
    - under_matched: people with fewer than 4 matches
    - unmatched_by_round: who was left out in each round
    - quality_stats: score distribution
    """
    # Track matches per person
    person_matches = {a.name: {"count": 0, "rounds": [], "scores": [], "as_helper": 0, "as_helped": 0} for a in attendees}

    # Go through all rounds and count matches
    for round_data in results["rounds"]:
        round_num = round_data["round"]
        round_type = round_data["type"]

        matched_this_round = set()

        for match in round_data["matches"]:
            if round_type == "complementary":
                helper_name = match["helper"]["name"]
                helped_name = match["helped"]["name"]
                score = match["score"]

                if helper_name in person_matches:
                    person_matches[helper_name]["count"] += 1
                    person_matches[helper_name]["rounds"].append(round_num)
                    person_matches[helper_name]["scores"].append(score)
                    person_matches[helper_name]["as_helper"] += 1
                    matched_this_round.add(helper_name)

                if helped_name in person_matches:
                    person_matches[helped_name]["count"] += 1
                    person_matches[helped_name]["rounds"].append(round_num)
                    person_matches[helped_name]["scores"].append(score)
                    person_matches[helped_name]["as_helped"] += 1
                    matched_this_round.add(helped_name)

                # Handle third person in trio
                if match.get("is_trio") and match.get("third_person"):
                    third_name = match["third_person"]["name"]
                    if third_name in person_matches:
                        person_matches[third_name]["count"] += 1
                        person_matches[third_name]["rounds"].append(round_num)
                        person_matches[third_name]["scores"].append(score)
                        person_matches[third_name]["as_helper"] += 0.5
                        person_matches[third_name]["as_helped"] += 0.5
                        matched_this_round.add(third_name)
            else:
                # Similarity round
                person_a = match["person_a"]["name"]
                person_b = match["person_b"]["name"]
                score = match["score"]

                for name in [person_a, person_b]:
                    if name in person_matches:
                        person_matches[name]["count"] += 1
                        person_matches[name]["rounds"].append(round_num)
                        person_matches[name]["scores"].append(score)
                        matched_this_round.add(name)

                # Handle third person in trio
                if match.get("is_trio") and match.get("person_c"):
                    third_name = match["person_c"]["name"]
                    if third_name in person_matches:
                        person_matches[third_name]["count"] += 1
                        person_matches[third_name]["rounds"].append(round_num)
                        person_matches[third_name]["scores"].append(score)
                        matched_this_round.add(third_name)

        # Track who was NOT matched this round
        round_data["_unmatched_names"] = [
            name for name in person_matches.keys()
            if name not in matched_this_round
        ]

    # Find under-matched people
    under_matched = []
    for name, data in person_matches.items():
        if data["count"] < 4:
            missing_rounds = [r for r in [1, 2, 3, 4] if r not in data["rounds"]]
            under_matched.append({
                "name": name,
                "match_count": data["count"],
                "missing_rounds": missing_rounds,
                "as_helper": data["as_helper"],
                "as_helped": data["as_helped"]
            })

    # Calculate quality stats
    all_scores = []
    for data in person_matches.values():
        all_scores.extend(data["scores"])

    quality_stats = {
        "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
        "min_score": min(all_scores) if all_scores else 0,
        "max_score": max(all_scores) if all_scores else 0,
        "low_quality_matches": len([s for s in all_scores if s < 50])
    }

    # Build unmatched by round summary
    unmatched_by_round = {}
    for round_data in results["rounds"]:
        round_num = round_data["round"]
        unmatched_names = round_data.get("_unmatched_names", [])
        if unmatched_names:
            unmatched_by_round[round_num] = unmatched_names

    return {
        "total_attendees": len(attendees),
        "fully_matched": len([p for p in person_matches.values() if p["count"] == 4]),
        "under_matched": under_matched,
        "unmatched_by_round": unmatched_by_round,
        "quality_stats": quality_stats,
        "per_person": person_matches
    }


def print_coverage_report(report: dict):
    """Print a human-readable coverage report."""
    print(f"\n{'='*60}")
    print(f"COVERAGE REPORT")
    print(f"{'='*60}")

    total = report["total_attendees"]
    fully_matched = report["fully_matched"]
    under_matched = report["under_matched"]

    print(f"\nAttendees: {total}")
    print(f"Fully matched (4/4 rounds): {fully_matched}")
    print(f"Under-matched: {len(under_matched)}")

    # Show under-matched people
    if under_matched:
        print(f"\n{'─'*60}")
        print("UNDER-MATCHED ATTENDEES:")
        print(f"{'─'*60}")
        for person in sorted(under_matched, key=lambda x: x["match_count"]):
            name = person["name"]
            count = person["match_count"]
            missing = person["missing_rounds"]
            helper = person["as_helper"]
            helped = person["as_helped"]
            print(f"\n  {name}")
            print(f"    Matches: {count}/4")
            print(f"    Missing rounds: {missing}")
            print(f"    Role balance: helper {helper}x, helped {helped}x")
    else:
        print(f"\n  Everyone was matched in all 4 rounds!")

    # Show unmatched by round
    if report["unmatched_by_round"]:
        print(f"\n{'─'*60}")
        print("UNMATCHED BY ROUND:")
        print(f"{'─'*60}")
        for round_num, names in report["unmatched_by_round"].items():
            print(f"\n  Round {round_num}: {', '.join(names)}")

    # Quality stats
    stats = report["quality_stats"]
    print(f"\n{'─'*60}")
    print("MATCH QUALITY:")
    print(f"{'─'*60}")
    print(f"  Average score: {stats['avg_score']:.1f}/100")
    print(f"  Range: {stats['min_score']} - {stats['max_score']}")
    print(f"  Low quality matches (<50): {stats['low_quality_matches']}")

    # Overall status
    print(f"\n{'='*60}")
    if len(under_matched) == 0:
        print("STATUS: All attendees fully matched!")
    else:
        print(f"STATUS: {len(under_matched)} attendee(s) need attention")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
#
# This runs when you execute: python matcher.py "path/to/luma.csv"
#
# Flags:
#   --interactive, -i : Step-by-step mode with pauses (default)
#   --auto            : Run all at once without pauses
#
# ============================================================================

def main():
    import sys

    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python matcher.py <path_to_luma_csv> [--auto]")
        print("\nExample:")
        print('  python matcher.py "MakersLounge Toronto Meetup #6 - Guests.csv"')
        print('  python matcher.py "MakersLounge Toronto Meetup #6 - Guests.csv" --auto')
        print("\nFlags:")
        print("  --auto    Run without pauses (default is interactive)")
        sys.exit(1)

    csv_path = sys.argv[1]
    interactive_mode = "--auto" not in sys.argv

    # ----------------------------------------------------------------
    # STEP 1: Load attendees from CSV
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 1: LOADING ATTENDEES")
    print(f"{'='*60}")
    print(f"\nFile: {csv_path}")

    all_attendees = parse_csv(csv_path)
    print(f"Loaded {len(all_attendees)} approved attendees")

    # ----------------------------------------------------------------
    # Check for incomplete profiles (no skills AND no needs)
    # ----------------------------------------------------------------
    complete_profiles = [a for a in all_attendees if a.has_complete_profile]
    incomplete_profiles = [a for a in all_attendees if not a.has_complete_profile]

    if incomplete_profiles:
        print(f"\n⚠ WARNING: {len(incomplete_profiles)} attendee(s) have INCOMPLETE PROFILES")
        print(f"  (No skills AND no needs listed - matching will be poor quality)")
        print()
        for p in incomplete_profiles:
            print(f"  • {p.name} - {p.project[:50]}{'...' if len(p.project) > 50 else ''}")

        if interactive_mode:
            print()
            choice = input("Skip these attendees? [Y/n]: ").strip().lower()
            if choice != 'n':
                attendees = complete_profiles
                print(f"\n→ Skipping {len(incomplete_profiles)} incomplete profiles")
                print(f"→ Proceeding with {len(attendees)} complete profiles")
            else:
                attendees = all_attendees
                print(f"\n→ Including all {len(attendees)} attendees (matches may be low quality)")
        else:
            # In auto mode, skip incomplete profiles by default
            attendees = complete_profiles
            print(f"\n→ Auto-skipping incomplete profiles, using {len(attendees)} attendees")
    else:
        attendees = all_attendees
        print(f"\n✓ All attendees have complete profiles")

    if interactive_mode:
        display_attendees_summary(attendees)
        wait_for_user("Step 1: Load Attendees")

    # ----------------------------------------------------------------
    # STEP 2: Pre-flight check
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 2: PRE-FLIGHT VALIDATION")
    print(f"{'='*60}")

    preflight_report = run_preflight_check(attendees)
    can_proceed = print_preflight_report(preflight_report)

    if not can_proceed:
        print("Cannot proceed with matching. Please fix the errors above.")
        sys.exit(1)

    # In interactive mode, allow user to abort if there are warnings
    if interactive_mode and preflight_report["warnings"]:
        choice = input("Continue with matching? [Y/n]: ").strip().lower()
        if choice == 'n':
            print("Matching cancelled by user.")
            sys.exit(0)

    # ----------------------------------------------------------------
    # STEP 3: Run all 4 rounds of matching
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 3: RUNNING MATCHING ROUNDS")
    print(f"{'='*60}")

    if interactive_mode:
        results = run_all_rounds_interactive(attendees, csv_path)
    else:
        results = run_all_rounds(attendees)

    # ----------------------------------------------------------------
    # STEP 4: Save results to JSON file
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 4: SAVING RESULTS")
    print(f"{'='*60}")

    # Simple output filenames in the same directory as the input
    output_dir = os.path.dirname(csv_path) or '.'
    output_path = os.path.join(output_dir, 'matches.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  → Saved: {output_path}")

    if interactive_mode:
        wait_for_user("Step 4: Save JSON")

    # ----------------------------------------------------------------
    # STEP 5: Export to CSV files (for printing/sharing)
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 5: EXPORTING TO CSV")
    print(f"{'='*60}")

    export_to_csv(results, output_dir)

    # ----------------------------------------------------------------
    # STEP 6: Generate and display coverage report
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 6: COVERAGE VALIDATION")
    print(f"{'='*60}")

    coverage_report = generate_coverage_report(results, attendees)
    print_coverage_report(coverage_report)

    # Save coverage report to JSON as well
    coverage_path = os.path.join(output_dir, 'coverage.json')
    with open(coverage_path, 'w') as f:
        json.dump(coverage_report, f, indent=2)
    print(f"  → Saved: {coverage_path}")

    # ----------------------------------------------------------------
    # STEP 7: AI Validation
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 7: AI VALIDATION")
    print(f"{'='*60}")

    ai_validation = run_ai_validation(results, attendees)
    print_ai_validation(ai_validation)

    # Save AI validation to JSON
    validation_path = os.path.join(output_dir, 'validation.json')
    with open(validation_path, 'w') as f:
        json.dump(ai_validation, f, indent=2)
    print(f"  → Saved: {validation_path}")

    # ----------------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"Total matches: {results['total_matches']}")
    print(f"Results saved to: {output_path}")

    for round_data in results['rounds']:
        print(f"\nRound {round_data['round']} ({round_data['type']}): {round_data['match_count']} matches")

    # Show coverage status in final summary
    if coverage_report["under_matched"]:
        print(f"\n⚠ WARNING: {len(coverage_report['under_matched'])} attendee(s) were under-matched")
        print(f"  See {coverage_path} for details")
    else:
        print(f"\n✓ All {coverage_report['total_attendees']} attendees matched in all 4 rounds")


# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()
