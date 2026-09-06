import json
from datetime import datetime
from urllib.parse import quote_plus

import requests
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types


# ============================================================
# AI MOOD MUSIC RECOMMENDER
# AI + Streamlit
# ============================================================

st.set_page_config(
    page_title="AI Mood Music Recommender",
    page_icon="🎵",
    layout="wide",
)

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .hero {
        padding: 2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #172554, #312e81, #581c87);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: .3rem;
    }

    .card {
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,.25);
        margin: .6rem 0;
        background: rgba(128,128,128,.06);
    }

    .song-title {
        font-size: 1.25rem;
        font-weight: 700;
    }

    .muted {
        opacity: .7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Music database
# ------------------------------------------------------------

SONGS = {
    "Happy": [
        {"title": "Katchi Sera", "artist": "Sai Abhyankkar"},
        {"title": "Aasa Kooda", "artist": "Sai Abhyankkar"},
        {"title": "Happy", "artist": "Pharrell Williams"},
        {"title": "On Top of the World", "artist": "Imagine Dragons"},
    ],
    "Sad": [
        {"title": "Someone Like You", "artist": "Adele"},
        {"title": "Fix You", "artist": "Coldplay"},
        {"title": "Yesterday", "artist": "The Beatles"},
        {"title": "The Night We Met", "artist": "Lord Huron"},
    ],
    "Energetic": [
        {"title": "Eye of the Tiger", "artist": "Survivor"},
        {"title": "Lose Yourself", "artist": "Eminem"},
        {"title": "Don't Stop Me Now", "artist": "Queen"},
        {"title": "Believer", "artist": "Imagine Dragons"},
    ],
    "Calm": [
        {"title": "Munbe Vaa", "artist": "A. R. Rahman"},
        {"title": "Weightless", "artist": "Marconi Union"},
        {"title": "Clair de Lune", "artist": "Claude Debussy"},
        {"title": "Come Away With Me", "artist": "Norah Jones"},
    ],
}


# ------------------------------------------------------------
# Historical user profiles
# ------------------------------------------------------------

USER_PROFILES = {
    "User A": {
        "Morning": "Energetic",
        "Afternoon": "Calm",
        "Evening": "Happy",
        "Night": "Calm",
    },
    "User B": {
        "Morning": "Happy",
        "Afternoon": "Energetic",
        "Evening": "Happy",
        "Night": "Calm",
    },
    "User C": {
        "Morning": "Energetic",
        "Afternoon": "Happy",
        "Evening": "Energetic",
        "Night": "Calm",
    },
}


# ------------------------------------------------------------
# Gemini configuration
# ------------------------------------------------------------

GEMINI_MODEL = "gemini-3.6-flash"

MOOD_VALUES = [
    "Happy",
    "Sad",
    "Energetic",
    "Calm",
]


def get_gemini_client():
    """Create a Gemini client from Streamlit secrets."""

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


# ------------------------------------------------------------
# Real AI story analysis
# ------------------------------------------------------------

def analyze_story_with_gemini(
    story: str,
    time_of_day: str,
    activity: str,
    historical_mood: str,
):
    """
    Uses Gemini to understand the user's story and return
    structured emotion/mood data.

    The four final recommendation moods remain:
    Happy, Sad, Energetic, Calm.
    """

    client = get_gemini_client()

    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Create .streamlit/secrets.toml and add your Gemini API key."
        )

    system_instruction = """
You are the AI emotion-analysis engine inside a college
AI-based music recommendation project.

Your job is to understand a user's free-form story.

Do NOT simply search for keywords. Understand the meaning,
context, emotional tone, and situation described by the user.

You may identify rich emotions such as:
Romantic, Excited, Happy, Sad, Relaxed, Calm, Lonely,
Stressed, Tired, Angry, Hopeful, Nostalgic, Confident,
Curious, Nervous, etc.

The final music mood MUST be exactly one of:
Happy, Sad, Energetic, Calm.

Decision rules:
1. The user's current story is the strongest signal.
2. Activity and time of day are supporting signals.
3. Historical mood is also a supporting signal.
4. If the current story conflicts with historical mood,
   prioritize the current story and mark mood_shift as true.
5. A romantic or positive story can map to Happy.
6. A peaceful or tired story can map to Calm.
7. An exciting/action-oriented story can map to Energetic.
8. A clearly negative/emotional story can map to Sad.
9. Do not diagnose medical or mental-health conditions.
10. Return only data matching the requested JSON schema.
"""

    prompt = f"""
Analyze this user's story.

USER STORY:
{story}

TIME OF DAY:
{time_of_day}

CURRENT ACTIVITY:
{activity}

HISTORICAL MOOD:
{historical_mood}

Return:
- primary_emotion
- secondary_emotions
- final_mood
- confidence from 0 to 100
- emotion_scores
- detected_signals
- reason
- mood_shift

The reason should explain the decision in simple language
that a college-project user can understand.
"""

    # Gemini 3.6 Flash uses the current Interactions API.
    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        system_instruction=system_instruction,
        generation_config={
            "thinking_level": "low",
        },
    )

    raw = getattr(interaction, "output_text", None)

    if not raw:
        raw_parts = []

        for step in getattr(interaction, "steps", []) or []:
            for content in getattr(step, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    raw_parts.append(text)

        raw = "".join(raw_parts)

    if not raw:
        raise RuntimeError("AI returned an empty response.")

    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.replace("```json", "", 1)
        raw = raw.replace("```", "", 1).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "AI returned a response that was not valid JSON. "
            "Please try again."
        ) from exc


    # Safety validation in application code.
    if result.get("final_mood") not in MOOD_VALUES:
        result["final_mood"] = "Calm"

    result["confidence"] = max(
        0,
        min(100, int(result.get("confidence", 50))),
    )

    return result


# ------------------------------------------------------------
# YouTube player
# ------------------------------------------------------------

def search_youtube_video(title: str, artist: str):
    """
    Find a real YouTube video ID using the YouTube Data API.

    Gemini decides the song. YouTube's API finds the actual video.
    The embeddable filter helps avoid videos that cannot play inside
    the Streamlit website.
    """
    try:
        api_key = st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        return None

    if not api_key:
        return None

    # Search for the official song first.
    queries = [
        f"{artist} {title} official",
        f"{title} {artist} official music video",
        f"{title} {artist} song",
    ]

    endpoint = "https://www.googleapis.com/youtube/v3/search"

    for query in queries:
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 5,
            "videoEmbeddable": "true",
            "videoSyndicated": "true",
            "regionCode": "IN",
            "key": api_key,
        }

        response = requests.get(
            endpoint,
            params=params,
            timeout=10,
        )

        if response.status_code != 200:
            continue

        data = response.json()

        for item in data.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            video_title = item.get("snippet", {}).get("title", "")

            if video_id:
                return {
                    "video_id": video_id,
                    "title": video_title,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                }

    return None


def youtube_player(video_id: str):
    """Embed a verified YouTube video inside Streamlit."""
    components.iframe(
        f"https://www.youtube.com/embed/{video_id}",
        height=430,
        scrolling=False,
    )


# ------------------------------------------------------------
# AI song recommendation layer
# ------------------------------------------------------------

def recommend_songs_with_gemini(
    story: str,
    analysis: dict,
    activity: str,
    time_of_day: str,
    previous_songs: list[str],
):
    """
    AI chooses the actual songs based on the user's story,
    detected emotions, mood and context.

    YouTube is only used afterwards to find the real playable video.
    """
    client = get_gemini_client()

    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Create .streamlit/secrets.toml and add your Gemini API key."
        )

    system_instruction = """
You are the music recommendation engine inside an AI mood music
recommendation project.

Your job is to recommend REAL songs that genuinely fit the user's
specific story and emotional situation.

Do NOT choose songs from a fixed list.
Do NOT recommend generic songs just because they match the final mood.
Understand the story, emotions, situation, language/context and activity.

You may recommend Tamil, English, Hindi or other songs when appropriate.
Prefer songs that are well-known and likely to have an official or
legitimate YouTube upload.

Avoid recommending songs already present in PREVIOUSLY RECOMMENDED SONGS.

Return exactly 5 song recommendations.

Return ONLY valid JSON in this structure:
{
  "songs": [
    {
      "title": "Song title",
      "artist": "Artist name",
      "why": "Short explanation of why this song fits the story"
    }
  ]
}
"""

    previous_text = ", ".join(previous_songs[-20:]) if previous_songs else "None"

    prompt = f"""
USER STORY:
{story}

AI EMOTION ANALYSIS:
Primary emotion: {analysis.get("primary_emotion", "")}
Secondary emotions: {", ".join(analysis.get("secondary_emotions", []))}
Final mood: {analysis.get("final_mood", "")}
Confidence: {analysis.get("confidence", 0)}%

TIME OF DAY:
{time_of_day}

CURRENT ACTIVITY:
{activity}

PREVIOUSLY RECOMMENDED SONGS:
{previous_text}

Recommend 5 different real songs that best fit THIS specific story.
Do not repeat any previous song.
For each song, explain briefly why it fits.
"""

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        system_instruction=system_instruction,
        generation_config={
            "thinking_level": "low",
        },
    )

    raw = getattr(interaction, "output_text", None)

    if not raw:
        raw_parts = []
        for step in getattr(interaction, "steps", []) or []:
            for content in getattr(step, "content", []) or []:
                value = getattr(content, "text", None)
                if value:
                    raw_parts.append(value)
        raw = "".join(raw_parts)

    if not raw:
        raise RuntimeError("AI returned an empty song recommendation response.")

    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.replace("```json", "", 1)
        raw = raw.replace("```", "", 1).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "AI returned invalid song recommendation JSON. Please try again."
        ) from exc

    songs = result.get("songs", [])

    if not isinstance(songs, list):
        raise RuntimeError("AI returned an invalid song list.")

    cleaned = []
    seen = set()

    for song in songs:
        title = str(song.get("title", "")).strip()
        artist = str(song.get("artist", "")).strip()
        why = str(song.get("why", "")).strip()

        if not title or not artist:
            continue

        key = f"{title.lower()}|{artist.lower()}"

        if key in seen:
            continue

        seen.add(key)

        cleaned.append(
            {
                "title": title,
                "artist": artist,
                "why": why,
            }
        )

    if not cleaned:
        raise RuntimeError("AI did not return usable song recommendations.")

    return cleaned[:5]


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Context")

    user = st.selectbox(
        "User",
        list(USER_PROFILES.keys()),
    )

    time_of_day = st.selectbox(
        "Time of Day",
        [
            "Morning",
            "Afternoon",
            "Evening",
            "Night",
        ],
    )

    activity = st.selectbox(
        "Current Activity",
        [
            "Relaxing",
            "Study",
            "Workout",
            "Commuting",
            "Party",
            "None / Not specified",
        ],
    )

    historical_mood = USER_PROFILES[user][time_of_day]

    st.divider()

    st.caption("Historical preference")

    st.info(
        f"{user} usually prefers **{historical_mood}** "
        f"during the {time_of_day.lower()}."
    )

    if st.button(
        "🗑️ Clear Session",
        use_container_width=True,
    ):
        st.session_state.history = []
        st.session_state.analysis = None
        st.session_state.recommendations = None
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🎵 AI Mood Music Recommender</h1>
        <p>
            Tell AI what's on your mind. AI understands the emotion,
            finds your mood and recommends songs.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Your story is the primary signal. Time of day, activity "
    "and historical preference are supporting signals."
)


# ============================================================
# STORY INPUT
# ============================================================

st.header("📝 Tell Me Your Story")

story = st.text_area(
    "What's on your mind?",
    placeholder=(
        "Example: I had a really good day today. I finished something "
        "I had been working on for a long time, and now I feel proud and relaxed..."
    ),
    height=180,
)

if st.button(
    "🧠 Analyze My Story with AI",
    type="primary",
    use_container_width=True,
):

    if not story.strip():

        st.warning(
            "Please tell Gemini a little about your day first."
        )

    else:

        with st.spinner(
            "🤖 AI is understanding your story..."
        ):

            try:

                analysis = analyze_story_with_gemini(
                    story=story,
                    time_of_day=time_of_day,
                    activity=activity,
                    historical_mood=historical_mood,
                )

                previous_songs = [
                    item.get("song", "")
                    for item in st.session_state.history
                    if item.get("song")
                ]

                # AI now chooses the actual songs based on the story.
                recommendations = recommend_songs_with_gemini(
                    story=story,
                    analysis=analysis,
                    activity=activity,
                    time_of_day=time_of_day,
                    previous_songs=previous_songs,
                )

                # YouTube only finds the real playable video for each
                # song selected by Gemini.
                for song in recommendations:
                    song["youtube"] = search_youtube_video(
                        song["title"],
                        song["artist"],
                    )

                st.session_state.analysis = analysis
                st.session_state.recommendations = recommendations

                st.session_state.history.append(
                    {
                        "time": datetime.now().strftime("%I:%M %p"),
                        "user": user,
                        "mood": analysis["final_mood"],
                        "emotion": analysis["primary_emotion"],
                        "activity": activity,
                        "time_of_day": time_of_day,
                        "song": f"{recommendations[0]['title']} — {recommendations[0]['artist']}",
                    }
                )

            except Exception as error:

                st.error(
                    "AI could not analyze the story."
                )
                st.code(str(error))


# ============================================================
# AI ANALYSIS
# ============================================================

analysis = st.session_state.analysis

if analysis:

    st.divider()

    st.header("🧠 AI Emotion Analysis")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Primary Emotion",
            analysis["primary_emotion"],
        )

    with c2:
        st.metric(
            "Final Music Mood",
            analysis["final_mood"],
        )

    with c3:
        st.metric(
            "AI Confidence",
            f'{analysis["confidence"]}%',
        )

    if analysis["secondary_emotions"]:
        st.write(
            "**Secondary emotions:** "
            + ", ".join(
                analysis["secondary_emotions"]
            )
        )

    st.subheader("💡 Why did AI choose this mood?")

    st.info(
        analysis["reason"]
    )

    if analysis["mood_shift"]:

        st.warning(
            f"🔄 **Mood Shift Detected:** Your historical mood "
            f"is **{historical_mood}**, but your current story "
            f"suggests **{analysis['final_mood']}**. "
            "The current story receives higher priority."
        )

    else:

        st.success(
            f"✅ Current story mood is consistent with your "
            f"historical **{historical_mood}** preference."
        )


# ============================================================
# EMOTION SCORES
# ============================================================

if analysis:

    st.subheader("📊 Emotion Signals")

    emotion_scores = analysis["emotion_scores"]

    ranked = sorted(
        emotion_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    shown = 0

    for emotion, score in ranked:

        if score <= 0:
            continue

        st.write(
            f"**{emotion} — {score}%**"
        )

        st.progress(
            min(100, int(score))
        )

        shown += 1

        if shown >= 6:
            break

    if analysis["detected_signals"]:

        st.subheader("🔍 What AI noticed")

        for signal in analysis["detected_signals"]:
            st.write(f"• {signal}")


# ============================================================
# SONG RECOMMENDATIONS
# ============================================================

recommendations = st.session_state.recommendations

if recommendations:

    st.divider()

    st.header("🎵 Personalized Playlist")

    # First recommendation is the main "Now Playing" result.
    now_playing = recommendations[0]

    st.subheader("▶️ Now Playing — AI's Top Recommendation")
    st.markdown(
        f"""
        <div class="card">
            <div class="song-title">
                🎵 {now_playing["title"]}
            </div>
            <div>
                Artist: {now_playing["artist"]}
            </div>
            <div class="muted">
                Selected from your story, emotion and context.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if now_playing.get("youtube"):
        youtube_player(now_playing["youtube"]["video_id"])
        st.caption(
            f"Playing YouTube result: {now_playing['youtube']['title']}"
        )
        st.link_button(
            "Open this song on YouTube ↗",
            now_playing["youtube"]["url"],
        )
    else:
        st.warning(
            "No embeddable YouTube result was found for this song. "
            "Add a YouTube API key in Streamlit secrets."
        )

    st.subheader("🎶 More Recommended Songs")

    for index, song in enumerate(
        recommendations[1:],
        start=2,
    ):
        with st.expander(
            f"🎵 Track {index}: {song['title']} — {song['artist']}"
        ):
            if song.get("why"):
                st.write(f"🧠 **Why Gemini chose it:** {song['why']}")

            if song.get("youtube"):
                youtube_player(song["youtube"]["video_id"])
                st.link_button(
                    "Open on YouTube ↗",
                    song["youtube"]["url"],
                )
            else:
                st.caption(
                    "No embeddable YouTube result found."
                )

    with st.expander("🔬 View AI Decision Pipeline"):

        st.code(
            f"""
USER STORY
    ↓
GEMINI AI
    ↓
Primary Emotion: {analysis["primary_emotion"]}
    ↓
Final Mood: {analysis["final_mood"]}
    ↓
Historical Mood: {historical_mood}
    ↓
Time: {time_of_day}
    ↓
Activity: {activity}
    ↓
Mood Shift: {analysis["mood_shift"]}
    ↓
PERSONALIZED PLAYLIST
    ↓
YOUTUBE PLAYER
            """,
            language="text",
        )


# ============================================================
# HISTORY
# ============================================================

st.divider()

st.header("📈 Mood History")

if st.session_state.history:

    for item in reversed(
        st.session_state.history[-10:]
    ):

        st.markdown(
            f"""
            <div class="card">
                <b>{item["time"]}</b> ·
                {item["user"]} ·
                <b>{item["mood"]}</b> ·
                {item["emotion"]} ·
                {item["activity"]}
                <br>
                🎵 {item["song"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

else:

    st.caption(
        "Your previous AI mood analyses will appear here."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Mood Music Recommender • AI • YouTube Data API • Streamlit"
)

with st.expander("⚙️ YouTube API setup"):
    st.write(
        "Add YOUTUBE_API_KEY to .streamlit/secrets.toml. "
        "The app uses YouTube's official search API to find a real "
        "embeddable video instead of guessing a YouTube URL."
    )
