"""Embeddings-based starter matcher with pre-synthesised audio cache.

Provides instant (~20-50ms) contextually-appropriate verbal responses
while the real LLM processes in the background.  Each starter has:
  - A spoken text ("Of course, let me look into that for you.")
  - A category tag (task_request, greeting, casual, ...)
  - A set of trigger phrases that represent user utterances this
    starter is a good match for

At startup:
  1. Embed all trigger phrases with a lightweight sentence model.
  2. (Optional) Pre-synthesise each starter as raw PCM audio via the
     TTS API, keyed by (voice, starter_index).  This is done lazily
     on first use of a given voice.

At runtime:
  1. Embed the user's utterance (~5-10ms on CPU).
  2. Cosine-similarity against all trigger embeddings.
  3. Return the best-matching starter text + pre-synthesised audio.

Falls back to keyword matching if the embedding model is unavailable.
"""

import asyncio
import os
import re
import random
from collections import deque
from typing import Optional

import httpx
import numpy as np
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

EMBEDDINGS_BASE_URL = os.environ.get(
    "EMBEDDINGS_BASE_URL", "https://openrouter.ai/api/v1",
)
EMBEDDINGS_MODEL = os.environ.get(
    "EMBEDDINGS_MODEL", "openai/text-embedding-3-large",
)
EMBEDDINGS_API_KEY = os.environ.get(
    "EMBEDDINGS_API_KEY",
    os.environ.get("OPENROUTER_API_KEY", ""),
)

# ---------------------------------------------------------------------------
# Starter library
# ---------------------------------------------------------------------------

STARTERS = [
    # =================================================================
    # GREETING — pure hello, no question attached
    # =================================================================
    {
        "text": "Hey there! Good to see you.",
        "category": "greeting",
        "triggers": [
            "hello", "hi", "hey", "howdy", "yo",
        ],
    },
    {
        "text": "Good to hear from you!",
        "category": "greeting",
        "triggers": [
            "hi there", "hey ron", "hello ron", "greetings",
            "nice to meet you", "hi ron",
        ],
    },
    {
        "text": "Hey, welcome! Glad you're here.",
        "category": "greeting",
        "triggers": [
            "good morning", "good afternoon", "good evening",
            "morning", "afternoon",
        ],
    },
    {
        "text": "Hey! Nice to talk to you.",
        "category": "greeting",
        "triggers": [
            "what's up", "what's going on", "hey what's up",
            "sup", "hiya",
        ],
    },

    # =================================================================
    # HOW_ARE_YOU — social question about Ron's state
    # =================================================================
    {
        "text": "I'm doing great, thanks for asking!",
        "category": "how_are_you",
        "triggers": [
            "how are you", "how are you doing", "how's it going",
            "how are you today", "how do you do",
        ],
    },
    {
        "text": "Doing well, thanks!",
        "category": "how_are_you",
        "triggers": [
            "you doing okay", "are you doing well",
            "how you doing", "what's new with you",
        ],
    },
    {
        "text": "I'm great! Thanks for asking.",
        "category": "how_are_you",
        "triggers": [
            "how's your day", "how's your day going",
            "having a good day", "how've you been",
        ],
    },
    {
        "text": "All good on my end!",
        "category": "how_are_you",
        "triggers": [
            "everything good", "you good", "all good",
            "are you alright", "you okay",
        ],
    },

    # =================================================================
    # TASK_REQUEST — user wants Ron to do something
    # =================================================================
    {
        "text": "On it, let me check that for you.",
        "category": "task_request",
        "triggers": [
            "check my plan", "look up my account", "what's my balance",
            "what is my data usage", "pull up my details",
        ],
    },
    {
        "text": "Let me pull that up.",
        "category": "task_request",
        "triggers": [
            "tell me about my package", "I need help with my account",
            "can you check my", "I want to see my", "show me my",
        ],
    },
    {
        "text": "Give me just a moment.",
        "category": "task_request",
        "triggers": [
            "run an analysis", "search for", "find me", "look up",
            "can you find", "do a search", "I need you to look",
        ],
    },
    {
        "text": "Working on that right now.",
        "category": "task_request",
        "triggers": [
            "train a model", "start training", "build a model",
            "run the pipeline", "start the analysis", "process the data",
        ],
    },
    {
        "text": "Let me get that started.",
        "category": "task_request",
        "triggers": [
            "change my plan", "switch my package", "upgrade my plan",
            "I want to change", "can you switch", "close my contract",
        ],
    },
    {
        "text": "I'll take care of that.",
        "category": "task_request",
        "triggers": [
            "set up alerts", "set data alerts", "notify me when",
            "can you set", "help me with roaming", "I'm traveling to",
        ],
    },
    {
        "text": "Let me look into that.",
        "category": "task_request",
        "triggers": [
            "buy a roaming pass", "purchase roaming", "activate roaming",
            "I need roaming for", "add an international package",
        ],
    },
    {
        "text": "One second, let me look into that.",
        "category": "task_request",
        "triggers": [
            "can you help me with", "I need to", "could you",
            "would you be able to", "I'd like to",
        ],
    },

    # =================================================================
    # STATUS_CHECK — user asking about ongoing work
    # =================================================================
    {
        "text": "Let me check on that.",
        "category": "status_check",
        "triggers": [
            "what's the status", "how far along", "is it done yet",
            "any update", "is it ready",
        ],
    },
    {
        "text": "Good question, let me see.",
        "category": "status_check",
        "triggers": [
            "how's it going with", "are you still working on",
            "progress update", "where are we",
        ],
    },
    {
        "text": "Let me see where we are.",
        "category": "status_check",
        "triggers": [
            "what's happening", "still running", "how long will it take",
            "when will it be done", "any progress",
        ],
    },

    # =================================================================
    # ANSWER — user replying to our question
    # =================================================================
    {
        "text": "Got it, thanks for that.",
        "category": "answer",
        "triggers": [
            "my number is", "the code is", "it's", "that would be",
            "here's my", "my phone number", "the account is",
        ],
    },
    {
        "text": "Okay, sounds good to me.",
        "category": "answer",
        "triggers": [
            "yes please", "go ahead", "sure", "yeah", "that's right",
            "correct", "yes that's correct", "yep",
        ],
    },
    {
        "text": "Perfect, let me work with that.",
        "category": "answer",
        "triggers": [
            "the answer is", "I'd like the", "I want the",
            "I'll go with", "I choose", "option", "I'll take",
        ],
    },
    {
        "text": "Thanks for that.",
        "category": "answer",
        "triggers": [
            "here you go", "there you go", "this is it",
            "that's the one", "right here",
        ],
    },

    # =================================================================
    # CASUAL — chitchat, humor, curiosity
    # =================================================================
    {
        "text": "Ha, that's a good one!",
        "category": "casual",
        "triggers": [
            "tell me a joke", "that's funny", "make me laugh",
            "say something funny", "you're funny",
        ],
    },
    {
        "text": "That's a great question.",
        "category": "casual",
        "triggers": [
            "what do you think about", "how does that work",
            "can you explain", "what's your opinion",
        ],
    },
    {
        "text": "Hmm, let me think.",
        "category": "casual",
        "triggers": [
            "what if", "hypothetically", "imagine", "suppose",
            "would you rather",
        ],
    },
    {
        "text": "Oh, that's really interesting!",
        "category": "casual",
        "triggers": [
            "did you know", "fun fact", "guess what",
            "I just found out", "apparently",
        ],
    },
    {
        "text": "Good question! Let me think about that.",
        "category": "casual",
        "triggers": [
            "I was wondering", "do you know", "is it true that",
            "I'm curious about", "have you heard of",
        ],
    },
    {
        "text": "That's interesting, tell me more.",
        "category": "casual",
        "triggers": [
            "that's interesting", "that's cool", "I didn't know that",
            "really", "wow", "no way",
        ],
    },

    # =================================================================
    # FAREWELL — user leaving
    # =================================================================
    {
        "text": "Have a great day!",
        "category": "farewell",
        "triggers": [
            "bye", "goodbye", "see you", "that's all",
            "I'm done", "that's everything",
        ],
    },
    {
        "text": "Glad I could help!",
        "category": "farewell",
        "triggers": [
            "thank you", "thanks for your help", "you've been helpful",
            "appreciate it", "thanks a lot",
        ],
    },
    {
        "text": "Take care, talk to you soon!",
        "category": "farewell",
        "triggers": [
            "cheers", "thanks bye", "later", "gotta go",
            "talk to you later", "catch you later",
        ],
    },
    {
        "text": "Anytime! Happy to help.",
        "category": "farewell",
        "triggers": [
            "thanks so much", "you're the best", "thank you so much",
            "you're awesome", "great help",
        ],
    },

    # =================================================================
    # ABOUT_YOU — user asking about Ron's capabilities
    # =================================================================
    {
        "text": "Great question! Let me tell you.",
        "category": "about_you",
        "triggers": [
            "what can you do", "what are your capabilities",
            "what can you help me with", "what do you do",
        ],
    },
    {
        "text": "Let me introduce myself.",
        "category": "about_you",
        "triggers": [
            "who are you", "tell me about yourself",
            "what are you", "what's your name", "introduce yourself",
        ],
    },
    {
        "text": "Good question! So basically,",
        "category": "about_you",
        "triggers": [
            "how do you work", "what kind of assistant are you",
            "are you an AI", "are you a bot", "are you real",
        ],
    },

    # =================================================================
    # COMPLAINT — user is frustrated or upset
    # =================================================================
    {
        "text": "I'm sorry to hear that.",
        "category": "complaint",
        "triggers": [
            "this isn't working", "I'm frustrated", "this is broken",
            "nothing works", "I've been waiting", "terrible service",
        ],
    },
    {
        "text": "I understand, and I'm here to help.",
        "category": "complaint",
        "triggers": [
            "I'm not happy", "this is unacceptable", "I want to complain",
            "your service is", "I'm disappointed", "why is this so",
        ],
    },
    {
        "text": "Let me help fix that.",
        "category": "complaint",
        "triggers": [
            "something went wrong", "there's an error", "I got charged",
            "wrong amount", "overcharged", "bill is wrong",
        ],
    },

    # =================================================================
    # AGREE — user agrees, confirms, or expresses enthusiasm
    # =================================================================
    {
        "text": "Absolutely, I couldn't agree more!",
        "category": "agree",
        "triggers": [
            "I agree", "that's right", "exactly", "totally",
            "absolutely", "you're right", "I think so too",
        ],
    },
    {
        "text": "Right? I feel the same way.",
        "category": "agree",
        "triggers": [
            "same here", "me too", "I feel that way too",
            "I know right", "for sure", "so true",
        ],
    },
    {
        "text": "Oh yes, I love that too!",
        "category": "agree",
        "triggers": [
            "I love that", "I love it", "I love going there",
            "that's my favorite", "I really enjoy", "it's so nice",
        ],
    },

    # =================================================================
    # SHARE — user sharing a fact, experience, or opinion
    # =================================================================
    {
        "text": "Oh, that's so cool!",
        "category": "share",
        "triggers": [
            "have you ever been", "I went to", "I visited",
            "have you been to", "I was at", "I once went",
        ],
    },
    {
        "text": "Oh wow, I didn't know that!",
        "category": "share",
        "triggers": [
            "did you know that", "they shot a movie", "there's a movie",
            "there's a famous", "it's actually", "the funny thing is",
        ],
    },
    {
        "text": "That sounds amazing!",
        "category": "share",
        "triggers": [
            "it's really beautiful", "the scenery is", "the views are",
            "it's a great place", "you should go", "it's worth visiting",
        ],
    },
    {
        "text": "I know what you mean!",
        "category": "share",
        "triggers": [
            "it's kind of", "it feels like", "there's something about",
            "the vibe is", "it's got this", "there's a certain",
        ],
    },

    # =================================================================
    # CORRECT — user correcting or clarifying something
    # =================================================================
    {
        "text": "Ah, I see what you mean.",
        "category": "correct",
        "triggers": [
            "no I mean", "no I'm talking about", "not that one",
            "actually I meant", "no no", "I was referring to",
        ],
    },
    {
        "text": "Oh right, my mistake!",
        "category": "correct",
        "triggers": [
            "that's not what I said", "I didn't mean that",
            "let me rephrase", "what I meant was", "not exactly",
        ],
    },

    # =================================================================
    # TELL_ME — user asking to tell a story or elaborate
    # =================================================================
    {
        "text": "Oh, great question! So...",
        "category": "tell_me",
        "triggers": [
            "tell me about", "tell me more", "what is it about",
            "can you tell me", "I want to hear about", "what happened",
        ],
    },
    {
        "text": "Glad you asked! Let me explain.",
        "category": "tell_me",
        "triggers": [
            "how does that work", "what does that mean",
            "explain that to me", "I don't understand",
            "what's the story behind", "I never heard of that",
        ],
    },
    {
        "text": "Let me fill you in!",
        "category": "tell_me",
        "triggers": [
            "go on", "keep going", "and then what",
            "what happened next", "tell me the rest",
            "I'm listening", "continue",
        ],
    },

    # =================================================================
    # SPOOKY — eerie, mysterious, supernatural topics
    # =================================================================
    {
        "text": "Oh yeah, that's pretty eerie!",
        "category": "spooky",
        "triggers": [
            "it's eerie", "it's creepy", "it's spooky",
            "there are ghosts", "stories about ghosts", "haunted",
        ],
    },
    {
        "text": "That gives me chills!",
        "category": "spooky",
        "triggers": [
            "that's scary", "that's terrifying", "that's creepy",
            "people disappear", "mysterious", "supernatural",
        ],
    },

    # =================================================================
    # NATURE — places, travel, outdoors
    # =================================================================
    {
        "text": "Oh, I love that area!",
        "category": "nature",
        "triggers": [
            "have you been to the beach", "the coast is",
            "the mountains are", "I love the outdoors",
            "I love hiking", "the trails are",
        ],
    },
    {
        "text": "That's a beautiful spot!",
        "category": "nature",
        "triggers": [
            "it's so pretty there", "the bay is", "the lake is",
            "the forest is", "I love the view", "the sunset there",
        ],
    },
    {
        "text": "Great taste! I know that place well.",
        "category": "nature",
        "triggers": [
            "further north", "further south", "up the coast",
            "down the coast", "in the mountains", "by the ocean",
        ],
    },

    # =================================================================
    # UNCLEAR — mumble, hesitation, retraction
    # =================================================================
    {
        "text": "Sorry, could you say that again?",
        "category": "unclear",
        "triggers": [
            "um", "uh", "hmm", "sorry what",
        ],
    },
    {
        "text": "No worries, take your time.",
        "category": "unclear",
        "triggers": [
            "I don't know", "never mind", "forget it",
            "actually no", "wait", "hold on",
        ],
    },
]

# ---------------------------------------------------------------------------
# Embedding engine (OpenAI-compatible /v1/embeddings API)
# ---------------------------------------------------------------------------

_trigger_embeddings: Optional[np.ndarray] = None
_trigger_to_starter: list[int] = []
_embeddings_ready = False

_RECENT_CAPACITY = 5
_recent_starters: deque[int] = deque(maxlen=_RECENT_CAPACITY)

_CLAUSE_SPLIT = re.compile(r"[.!?;]+\s+|\s*,\s+(?:and|but|so|what|how|can|could|would|tell|do)\s+", re.IGNORECASE)


def _embed_sync(texts: list[str]) -> Optional[np.ndarray]:
    """Call the OpenAI-compatible embeddings endpoint synchronously."""
    import httpx as _httpx

    url = EMBEDDINGS_BASE_URL.rstrip("/")
    if not url.endswith("/embeddings"):
        url += "/embeddings"

    payload: dict = {"input": texts}
    if EMBEDDINGS_MODEL:
        payload["model"] = EMBEDDINGS_MODEL

    headers = {"Content-Type": "application/json"}
    if EMBEDDINGS_API_KEY and EMBEDDINGS_API_KEY != "not-needed":
        headers["Authorization"] = f"Bearer {EMBEDDINGS_API_KEY}"

    try:
        resp = _httpx.post(url, json=payload, headers=headers, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()

        embeddings = [None] * len(texts)
        for item in data["data"]:
            embeddings[item["index"]] = item["embedding"]

        matrix = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms
        return matrix

    except Exception as e:
        logger.error(f"[starters] Embeddings API call failed: {e}")
        return None


async def _embed_async(texts: list[str]) -> Optional[np.ndarray]:
    """Call the embeddings endpoint asynchronously (for runtime matching)."""
    url = EMBEDDINGS_BASE_URL.rstrip("/")
    if not url.endswith("/embeddings"):
        url += "/embeddings"

    payload: dict = {"input": texts}
    if EMBEDDINGS_MODEL:
        payload["model"] = EMBEDDINGS_MODEL

    headers = {"Content-Type": "application/json"}
    if EMBEDDINGS_API_KEY and EMBEDDINGS_API_KEY != "not-needed":
        headers["Authorization"] = f"Bearer {EMBEDDINGS_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        embeddings = [None] * len(texts)
        for item in data["data"]:
            embeddings[item["index"]] = item["embedding"]

        matrix = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms
        return matrix

    except Exception as e:
        logger.error(f"[starters] Async embeddings call failed: {e}")
        return None


def build_index():
    """Pre-compute trigger embeddings at startup via the embeddings API."""
    global _trigger_embeddings, _trigger_to_starter, _embeddings_ready

    all_triggers: list[str] = []
    _trigger_to_starter = []

    for idx, starter in enumerate(STARTERS):
        for trigger in starter["triggers"]:
            all_triggers.append(trigger)
            _trigger_to_starter.append(idx)

    logger.info(
        f"[starters] Embedding {len(all_triggers)} triggers via "
        f"{EMBEDDINGS_BASE_URL}..."
    )
    _trigger_embeddings = _embed_sync(all_triggers)

    if _trigger_embeddings is not None:
        _embeddings_ready = True
        logger.info(
            f"[starters] Index built: {len(all_triggers)} triggers "
            f"→ {len(STARTERS)} starters "
            f"(dim={_trigger_embeddings.shape[1]})"
        )
    else:
        _embeddings_ready = False
        logger.warning("[starters] Index built (keyword-only fallback)")


def _extract_last_clause(utterance: str) -> Optional[str]:
    """Return the last meaningful clause if the utterance has multiple parts.

    Splits on sentence boundaries and conjunctions that introduce a new
    intent (e.g. "My day is fine. What can you do?" → "What can you do?").
    Returns None if the utterance is a single clause.
    """
    parts = _CLAUSE_SPLIT.split(utterance.strip())
    parts = [p.strip() for p in parts if p and len(p.strip()) > 5]
    if len(parts) >= 2:
        return parts[-1]
    return None


def _score_utterance(text: str) -> Optional[np.ndarray]:
    """Embed a single text and return the similarity scores against all triggers."""
    if not _embeddings_ready or _trigger_embeddings is None:
        return None
    query_emb = _embed_sync([text.strip()])
    if query_emb is None:
        return None
    return (query_emb @ _trigger_embeddings.T)[0]


def match(
    utterance: str,
    threshold: float = 0.35,
    top_k: int = 8,
) -> Optional[dict]:
    """Find a matching starter for a user utterance with rotation.

    Improvements over naive best-match:
    - When the utterance has multiple clauses, also scores the last
      clause independently to capture the actual user intent.
    - Tracks recently used starters and avoids them when alternatives
      exist.

    Returns {"text": ..., "category": ..., "index": ...} or None.
    """
    if not utterance or not utterance.strip():
        return None

    if not _embeddings_ready or _trigger_embeddings is None:
        return _keyword_fallback(utterance)

    flat_scores = _score_utterance(utterance)
    if flat_scores is None:
        return _keyword_fallback(utterance)

    last_clause = _extract_last_clause(utterance)
    if last_clause:
        clause_scores = _score_utterance(last_clause)
        if clause_scores is not None:
            flat_scores = np.maximum(flat_scores, clause_scores)
            logger.debug(
                f"[starters] Multi-clause detected, also scored: '{last_clause[:40]}'"
            )

    top_indices = np.argsort(flat_scores)[::-1][:top_k]

    candidates: dict[int, float] = {}
    for idx in top_indices:
        score = float(flat_scores[idx])
        if score < threshold:
            break
        starter_idx = _trigger_to_starter[idx]
        if starter_idx not in candidates or score > candidates[starter_idx]:
            candidates[starter_idx] = score

    if not candidates:
        best_score = float(flat_scores[top_indices[0]]) if len(top_indices) > 0 else 0.0
        logger.debug(
            f"[starters] No match above threshold "
            f"(best={best_score:.2f} < {threshold})"
        )
        return None

    fresh = {k: v for k, v in candidates.items() if k not in _recent_starters}
    pool = fresh if fresh else candidates

    chosen_idx = random.choice(list(pool.keys()))
    chosen_score = pool[chosen_idx]
    starter = STARTERS[chosen_idx]

    _recent_starters.append(chosen_idx)

    logger.info(
        f"[starters] Match: '{utterance[:40]}' → "
        f"'{starter['text'][:40]}' ({starter['category']}) "
        f"score={chosen_score:.2f} "
        f"(from {len(candidates)} candidate(s), "
        f"{len(candidates) - len(fresh)} recently used)"
    )
    return {
        "text": starter["text"],
        "category": starter["category"],
        "index": chosen_idx,
        "score": chosen_score,
    }


def _keyword_fallback(utterance: str) -> Optional[dict]:
    """Simple keyword overlap fallback when embeddings are unavailable."""
    lower = utterance.lower()
    best_starter = None
    best_overlap = 0

    for idx, starter in enumerate(STARTERS):
        for trigger in starter["triggers"]:
            trigger_words = set(trigger.lower().split())
            utterance_words = set(lower.split())
            overlap = len(trigger_words & utterance_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_starter = (idx, starter)

    if best_starter and best_overlap >= 1:
        idx, starter = best_starter
        logger.info(
            f"[starters] Keyword fallback: '{utterance[:40]}' → "
            f"'{starter['text'][:40]}' (overlap={best_overlap})"
        )
        return {
            "text": starter["text"],
            "category": starter["category"],
            "index": idx,
            "score": best_overlap / 5.0,
        }

    return None


# ---------------------------------------------------------------------------
# Pre-synthesised audio cache
# ---------------------------------------------------------------------------

WAV_HEADER_SIZE = 44
SAMPLE_RATE = 24000

_audio_cache: dict[tuple[str, int], bytes] = {}
_synth_lock = asyncio.Lock()


async def presynthesize(
    voice: str,
    tts_server: str = "http://192.168.7.163:8100",
    language: str = "English",
):
    """Pre-synthesise all starters for a given voice.

    Stores raw PCM bytes (no WAV header) keyed by (voice, starter_index).
    Call at pipeline startup or when the voice changes.
    """
    async with _synth_lock:
        cached = sum(1 for k in _audio_cache if k[0] == voice)
        if cached >= len(STARTERS):
            logger.info(f"[starters] Audio cache already complete for voice={voice}")
            return

        logger.info(
            f"[starters] Pre-synthesising {len(STARTERS)} starters "
            f"for voice={voice}..."
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            for idx, starter in enumerate(STARTERS):
                key = (voice, idx)
                if key in _audio_cache:
                    continue
                try:
                    resp = await client.post(
                        f"{tts_server}/v1/tts/stream",
                        json={
                            "text": starter["text"],
                            "language": language,
                            "voice": voice,
                        },
                    )
                    resp.raise_for_status()
                    raw_audio = resp.content
                    if len(raw_audio) > WAV_HEADER_SIZE:
                        raw_audio = raw_audio[WAV_HEADER_SIZE:]
                    _audio_cache[key] = raw_audio
                except Exception as e:
                    logger.warning(
                        f"[starters] Failed to synthesise starter {idx} "
                        f"for voice={voice}: {e}"
                    )

        cached = sum(1 for k in _audio_cache if k[0] == voice)
        total_kb = sum(len(v) for k, v in _audio_cache.items() if k[0] == voice) / 1024
        logger.info(
            f"[starters] Pre-synthesis complete: {cached}/{len(STARTERS)} "
            f"starters for voice={voice} ({total_kb:.0f} KB)"
        )


def get_audio(voice: str, starter_index: int) -> Optional[bytes]:
    """Get pre-synthesised raw PCM audio for a starter.

    Returns None if not yet synthesised for this voice.
    """
    return _audio_cache.get((voice, starter_index))


def get_audio_frames(
    voice: str,
    starter_index: int,
    chunk_samples: int = 4800,
) -> list[bytes]:
    """Split pre-synthesised audio into frame-sized chunks.

    Each chunk is `chunk_samples` samples of 16-bit PCM (= chunk_samples * 2 bytes).
    Default chunk_samples=4800 = 200ms at 24kHz.
    """
    raw = get_audio(voice, starter_index)
    if raw is None:
        return []

    bytes_per_chunk = chunk_samples * 2
    return [raw[i:i + bytes_per_chunk] for i in range(0, len(raw), bytes_per_chunk)]
