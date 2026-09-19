import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Spotify Stream Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load("spotify_stream_prediction_model.pkl")
    preprocessor = joblib.load("spotify_preprocessor.pkl")
    return model, preprocessor


try:
    model, preprocessor = load_model()

except Exception as e:

    st.error("Model files could not be loaded.")
    st.code(str(e))
    st.stop()


# =========================================================
# SIMPLE CLEAN CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.title("🎵 Spotify Stream Predictor")

st.caption(
    "Estimate how many streams a track could receive using Machine Learning."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🎧 About the Model")

    st.write(
        "This application uses an XGBoost regression model "
        "trained on historical Spotify track data."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** XGBoost Regressor")
    st.write("**Training Period:** 2020–2024")
    st.write("**Evaluation Year:** 2025")
    st.write("**Target:** Log Stream Count")
    st.write("**R² Score:** 0.5350")
    st.write("**MAE:** 71,353 streams")

    st.divider()

    st.info(
        "The model predicts an estimated stream count "
        "based on the characteristics of the track."
    )

    st.caption(
        "The prediction is an estimate based on historical data "
        "and does not guarantee actual future streams."
    )


# =========================================================
# 1. SONG INFORMATION
# =========================================================

st.header("1️⃣ Song Information")

st.caption(
    "Enter the basic information about the track."
)

with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:

        genre = st.selectbox(
            "Genre",
            [
                "Pop",
                "Rock",
                "Hip-Hop",
                "Rap",
                "R&B",
                "Electronic",
                "Dance",
                "Classical",
                "Jazz",
                "Country",
                "Metal",
                "Alternative",
                "Indie",
                "Reggae",
                "Other"
            ]
        )

        country = st.text_input(
            "Country Code",
            value="IN",
            help="Example: IN, US, GB, CA"
        ).upper()

        label = st.text_input(
            "Record Label",
            value="Independent"
        )

    with col2:

        duration_minutes = st.number_input(
            "Track Duration (minutes)",
            min_value=0.5,
            max_value=15.0,
            value=3.5,
            step=0.1
        )

        duration_ms = int(duration_minutes * 60 * 1000)

        popularity = st.slider(
            "Spotify Popularity",
            min_value=0,
            max_value=100,
            value=65,
            help="Spotify popularity score from 0 to 100."
        )


# =========================================================
# 2. AUDIO FEATURES
# =========================================================

st.header("2️⃣ Audio Features")

st.caption(
    "Describe the musical and audio characteristics of the track."
)

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        danceability = st.slider(
            "Danceability",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.01,
            help="How suitable the track is for dancing."
        )

        energy = st.slider(
            "Energy",
            min_value=0.0,
            max_value=1.0,
            value=0.80,
            step=0.01,
            help="Perceived intensity and activity of the track."
        )

    with col2:

        instrumentalness = st.slider(
            "Instrumentalness",
            min_value=0.0,
            max_value=1.0,
            value=0.00,
            step=0.01,
            help="Likelihood that the track contains no vocals."
        )

        tempo = st.number_input(
            "Tempo (BPM)",
            min_value=40.0,
            max_value=240.0,
            value=120.0,
            step=1.0
        )

    with col3:

        loudness = st.number_input(
            "Loudness (dB)",
            min_value=-60.0,
            max_value=5.0,
            value=-5.0,
            step=0.1
        )

        key = st.slider(
            "Musical Key",
            min_value=0,
            max_value=11,
            value=5
        )

        mode = st.selectbox(
            "Mode",
            [0, 1],
            format_func=lambda x:
                "Minor" if x == 0 else "Major"
        )


# =========================================================
# 3. RELEASE INFORMATION
# =========================================================

st.header("3️⃣ Release Information")

st.caption(
    "Select the release date. Other date-related features "
    "will be calculated automatically."
)

with st.container(border=True):

    release_date = st.date_input(
        "Release Date",
        value=date(2025, 6, 20)
    )

    release_year = release_date.year

    release_month = release_date.month

    release_day_of_week = release_date.strftime("%A")

    release_quarter = (
        f"Q{((release_month - 1) // 3) + 1}"
    )

    is_weekend_release = (
        release_date.weekday() >= 5
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Year",
            release_year
        )

    with col2:

        st.metric(
            "Month",
            release_month
        )

    with col3:

        st.metric(
            "Day",
            release_day_of_week
        )

    with col4:

        st.metric(
            "Weekend",
            "Yes" if is_weekend_release else "No"
        )

    st.write(
        f"**Release Quarter:** {release_quarter}"
    )


# =========================================================
# 4. ARTIST INFORMATION
# =========================================================

st.header("4️⃣ Artist & Track Details")

st.caption(
    "Enter additional information used by the model."
)

with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:

        artist_track_count = st.number_input(
            "Artist Track Count",
            min_value=1,
            max_value=1000,
            value=5,
            step=1,
            help=(
                "Number of tracks associated with the artist "
                "in the dataset."
            )
        )

    with col2:

        explicit = st.checkbox(
            "Explicit Content",
            value=False
        )

        is_explicit_bool = explicit

    st.write("")

    upbeat_score = st.slider(
        "Upbeat Score",
        min_value=0.0,
        max_value=1.0,
        value=0.78,
        step=0.01,
        help=(
            "Feature used by the model to represent "
            "the upbeat characteristics of the track."
        )
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.divider()

st.header("🚀 Prediction")

st.caption(
    "Review your inputs and click the button to estimate streams."
)

predict_button = st.button(
    "🎵 Predict Spotify Streams",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        song = pd.DataFrame([{

            "genre": genre,
            "duration_ms": duration_ms,
            "popularity": popularity,
            "danceability": danceability,
            "energy": energy,
            "key": key,
            "loudness": loudness,
            "mode": mode,
            "instrumentalness": instrumentalness,
            "tempo": tempo,
            "country": country,
            "explicit": explicit,
            "label": label,
            "release_year": release_year,
            "release_month": release_month,
            "release_day_of_week": release_day_of_week,
            "duration_minutes": duration_minutes,
            "is_explicit_bool": is_explicit_bool,
            "release_quarter": release_quarter,
            "is_weekend_release": is_weekend_release,
            "upbeat_score": upbeat_score,
            "artist_track_count": artist_track_count

        }])

        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        song_processed = preprocessor.transform(song)

        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        predicted_log = model.predict(
            song_processed
        )[0]

        predicted_streams = np.expm1(
            predicted_log
        )

        predicted_streams = max(
            0,
            predicted_streams
        )

        # -------------------------------------------------
        # SUCCESS
        # -------------------------------------------------

        st.success(
            "✅ Prediction completed successfully!"
        )

        st.divider()

        # -------------------------------------------------
        # MAIN RESULT
        # -------------------------------------------------

        st.subheader("🎯 Estimated Spotify Streams")

        st.metric(
            label="Predicted Stream Count",
            value=f"{predicted_streams:,.0f}"
        )

        # -------------------------------------------------
        # SHORT FORMAT
        # -------------------------------------------------

        if predicted_streams >= 1_000_000:

            short_value = (
                f"{predicted_streams / 1_000_000:.2f} Million"
            )

        elif predicted_streams >= 1_000:

            short_value = (
                f"{predicted_streams / 1_000:.1f}K"
            )

        else:

            short_value = (
                f"{predicted_streams:,.0f}"
            )

        st.info(
            f"📈 Approximate result: **{short_value} streams**"
        )

        # -------------------------------------------------
        # RESULT METRICS
        # -------------------------------------------------

        st.subheader("📊 Result Overview")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Streams",
                f"{predicted_streams:,.0f}"
            )

        with col2:

            st.metric(
                "Thousands",
                f"{predicted_streams / 1_000:,.1f}K"
            )

        with col3:

            st.metric(
                "Millions",
                f"{predicted_streams / 1_000_000:.2f}M"
            )

        # -------------------------------------------------
        # TRACK SUMMARY
        # -------------------------------------------------

        st.divider()

        st.subheader("📝 Track Summary")

        col1, col2 = st.columns(2)

        with col1:

            st.write(f"**Genre:** {genre}")

            st.write(f"**Country:** {country}")

            st.write(f"**Record Label:** {label}")

            st.write(
                f"**Popularity:** {popularity}/100"
            )

            st.write(
                f"**Duration:** {duration_minutes:.1f} minutes"
            )

            st.write(
                f"**Explicit:** "
                f"{'Yes' if explicit else 'No'}"
            )

        with col2:

            st.write(
                f"**Danceability:** {danceability:.2f}"
            )

            st.write(
                f"**Energy:** {energy:.2f}"
            )

            st.write(
                f"**Tempo:** {tempo:.0f} BPM"
            )

            st.write(
                f"**Release Date:** "
                f"{release_date.strftime('%d %B %Y')}"
            )

            st.write(
                f"**Release Quarter:** "
                f"{release_quarter}"
            )

            st.write(
                f"**Artist Track Count:** "
                f"{artist_track_count}"
            )

        # -------------------------------------------------
        # HOW IT WORKS
        # -------------------------------------------------

        st.divider()

        with st.expander(
            "🔍 How does this prediction work?"
        ):

            st.write(
                "The model takes the track information you "
                "entered and analyzes patterns learned from "
                "historical Spotify data."
            )

            st.write(
                "An XGBoost regression model first predicts "
                "the log-transformed stream count."
            )

            st.write(
                "The prediction is then converted back to "
                "the estimated number of Spotify streams."
            )

            st.warning(
                "This is an estimate based on historical "
                "patterns. Actual streams can be different."
            )

    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.code(str(e))

        st.warning(
            "Please check that the model and preprocessor "
            "files are compatible with your installed libraries."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Spotify Stream Prediction System • "
    "XGBoost Machine Learning • "
    "Historical Spotify Data"
)