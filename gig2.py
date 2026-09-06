import json
from datetime import datetime
from urllib.parse import quote_plus

import requests
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# CONFIG
# =========================================================

OPENROUTER_MODEL = "openrouter/free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Mood Music",
    page_icon="🎵",
    layout="wide"
)


# =========================================================
# OPENROUTER AI
# =========================================================

def get_openrouter_key():

    try:
        return st.secrets["OPENROUTER_API_KEY"]

    except Exception:
        st.error("OPENROUTER_API_KEY is missing from Streamlit Secrets.")
        st.stop()


def ask_ai(prompt):

    api_key = get_openrouter_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io/",
        "X-Title": "AI Mood Music Recommender"
    }

    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an AI music recommendation assistant. "
                    "Analyze emotions carefully and recommend suitable real songs."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=data,
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(
            f"OpenRouter error {response.status_code}: "
            f"{response.text}"
        )

    result = response.json()

    return result["choices"][0]["message"]["content"]


# =========================================================
# JSON CLEANER
# =========================================================

def clean_json(text):

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return json.loads(text)


# =========================================================
# STORY ANALYSIS
# =========================================================

def analyze_story(story, time_of_day, activity, historical_mood):

    prompt = f"""
Analyze the following person's story and determine their current emotional state.

STORY:
{story}

TIME OF DAY:
{time_of_day}

ACTIVITY:
{activity}

PREVIOUS MOOD:
{historical_mood}

Return ONLY valid JSON.

Use this exact structure:

{{
    "primary_emotion": "emotion",
    "secondary_emotions": ["emotion1", "emotion2"],
    "final_mood": "Happy",
    "confidence": 0.85,
    "emotion_scores": {{
        "Happy": 0.0,
        "Sad": 0.0,
        "Energetic": 0.0,
        "Calm": 0.0
    }},
    "detected_signals": ["signal1", "signal2"],
    "reason": "short explanation",
    "mood_shift": "short explanation"
}}

final_mood MUST be one of:

Happy
Sad
Energetic
Calm

confidence must be between 0 and 1.

emotion_scores must contain values between 0 and 1.
"""

    result = ask_ai(prompt)

    return clean_json(result)


# =========================================================
# AI SONG RECOMMENDATIONS
# =========================================================

def recommend_songs(
    story,
    analysis,
    activity,
    time_of_day,
    previous_songs
):

    prompt = f"""
You are an intelligent AI music recommendation system.

Recommend 5 REAL songs based on the user's current emotional state.

USER STORY:
{story}

AI EMOTION ANALYSIS:
{json.dumps(analysis, indent=2)}

ACTIVITY:
{activity}

TIME:
{time_of_day}

PREVIOUSLY RECOMMENDED SONGS:
{previous_songs}

Important requirements:

1. Recommend REAL songs that actually exist.
2. Do not invent songs.
3. Match the emotional situation, not just the mood label.
4. Consider the activity and time.
5. Avoid repeating previous songs.
6. Mix popular and highly suitable songs.
7. Include the artist.
8. Explain briefly why each song fits.
9. Return ONLY valid JSON.

Use exactly this structure:

{{
    "songs": [
        {{
            "title": "Song Title",
            "artist": "Artist Name",
            "why": "Why this song fits the user's current emotional state"
        }},
        {{
            "title": "Song Title",
            "artist": "Artist Name",
            "why": "Why this song fits"
        }},
        {{
            "title": "Song Title",
            "artist": "Artist Name",
            "why": "Why this song fits"
        }},
        {{
            "title": "Song Title",
            "artist": "Artist Name",
            "why": "Why this song fits"
        }},
        {{
            "title": "Song Title",
            "artist": "Artist Name",
            "why": "Why this song fits"
        }}
    ]
}}
"""

    result = ask_ai(prompt)

    return clean_json(result)


# =========================================================
# YOUTUBE SEARCH
# =========================================================

def search_youtube_video(title, artist):

    try:

        api_key = st.secrets["YOUTUBE_API_KEY"]

    except Exception:

        return None

    queries = [
        f"{artist} {title} official",
        f"{title} {artist} official music video",
        f"{title} {artist} song"
    ]

    for query in queries:

        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 5,
            "videoEmbeddable": "true",
            "videoSyndicated": "true",
            "regionCode": "IN",
            "key": api_key
        }

        try:

            response = requests.get(
                YOUTUBE_SEARCH_URL,
                params=params,
                timeout=20
            )

            if response.status_code != 200:
                continue

            data = response.json()

            items = data.get("items", [])

            if items:

                video_id = items[0]["id"]["videoId"]

                return {
                    "video_id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": items[0]["snippet"]["title"]
                }

        except Exception:
            continue

    return None


# =========================================================
# YOUTUBE PLAYER
# =========================================================

def youtube_player(video_id):

    components.iframe(
        f"https://www.youtube.com/embed/{video_id}",
        height=420
    )


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "songs" not in st.session_state:
    st.session_state.songs = []

if "analysis" not in st.session_state:
    st.session_state.analysis = None


# =========================================================
# HEADER
# =========================================================

st.title("🎵 AI Mood Music")

st.write(
    "Tell the AI what's on your mind. "
    "It analyzes your emotions and creates a personalized playlist."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Your Context")

user = st.sidebar.selectbox(
    "User",
    ["User A", "User B", "User C"]
)

time_of_day = st.sidebar.selectbox(
    "Time of Day",
    [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]
)

activity = st.sidebar.selectbox(
    "What are you doing?",
    [
        "Relaxing",
        "Studying",
        "Working",
        "Workout",
        "Travelling",
        "Sleeping",
        "Free time"
    ]
)

historical_mood = st.sidebar.selectbox(
    "Previous Mood",
    [
        "Happy",
        "Sad",
        "Energetic",
        "Calm"
    ]
)


# =========================================================
# STORY INPUT
# =========================================================

story = st.text_area(
    "What's on your mind?",
    placeholder=(
        "I had a really good day today. "
        "I finished something I had been working on for a long time, "
        "and now I feel proud and relaxed..."
    ),
    height=150
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "✨ Analyze My Mood",
    use_container_width=True
):

    if not story.strip():

        st.warning("Please tell me what's on your mind first.")

    else:

        with st.spinner("AI is understanding your story..."):

            try:

                analysis = analyze_story(
                    story,
                    time_of_day,
                    activity,
                    historical_mood
                )

                st.session_state.analysis = analysis

            except Exception as e:

                st.error(f"AI analysis failed: {e}")

                st.stop()


        # =================================================
        # AI ANALYSIS
        # =================================================

        st.subheader("🧠 AI Mood Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Primary Emotion",
                analysis["primary_emotion"]
            )

        with col2:
            st.metric(
                "Final Mood",
                analysis["final_mood"]
            )

        with col3:
            st.metric(
                "Confidence",
                f"{analysis['confidence'] * 100:.0f}%"
            )

        st.write(
            "**Why AI thinks this:**",
            analysis["reason"]
        )

        st.write(
            "**Mood shift:**",
            analysis["mood_shift"]
        )


        # =================================================
        # EMOTION SCORES
        # =================================================

        st.subheader("📊 Emotion Scores")

        scores = analysis["emotion_scores"]

        for emotion, score in scores.items():

            st.write(
                f"**{emotion} — {score * 100:.0f}%**"
            )

            st.progress(float(score))


        # =================================================
        # DETECTED SIGNALS
        # =================================================

        st.subheader("🔎 Story Signals")

        for signal in analysis["detected_signals"]:

            st.write(f"• {signal}")


        # =================================================
        # AI SONG RECOMMENDATION
        # =================================================

        previous_songs = [
            item["title"]
            for item in st.session_state.history
        ]

        with st.spinner("AI is choosing songs for you..."):

            try:

                recommendation = recommend_songs(
                    story,
                    analysis,
                    activity,
                    time_of_day,
                    previous_songs
                )

                songs = recommendation["songs"]

                st.session_state.songs = songs

            except Exception as e:

                st.error(
                    f"Song recommendation failed: {e}"
                )

                st.stop()


        # =================================================
        # PLAYLIST
        # =================================================

        st.subheader("🎧 Your AI Playlist")

        for index, song in enumerate(songs):

            title = song["title"]
            artist = song["artist"]
            why = song["why"]

            with st.container():

                st.markdown(
                    f"### {index + 1}. {title}"
                )

                st.write(
                    f"**Artist:** {artist}"
                )

                youtube = search_youtube_video(
                    title,
                    artist
                )

                if youtube:

                    youtube_player(
                        youtube["video_id"]
                    )

                else:

                    st.info(
                        "YouTube video could not be found."
                    )

                st.write(
                    f"💡 **Why AI chose it:** {why}"
                )

                st.divider()


                # Save history

                if not any(
                    h["title"] == title
                    for h in st.session_state.history
                ):

                    st.session_state.history.append({
                        "title": title,
                        "artist": artist,
                        "time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M"
                        )
                    })


# =========================================================
# HISTORY
# =========================================================

if st.session_state.history:

    st.subheader("🕘 Listening History")

    for item in reversed(
        st.session_state.history[-10:]
    ):

        st.write(
            f"🎵 **{item['title']}** — "
            f"{item['artist']} "
            f"({item['time']})"
        )


# =========================================================
# PIPELINE
# =========================================================

st.divider()

st.subheader("⚙️ AI Decision Pipeline")

st.write(
    "Story → Emotion Detection → Context Analysis → "
    "AI Recommendation → YouTube Search → Music Playback"
)
