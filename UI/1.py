import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import io
from pydub import AudioSegment
import time
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

st.set_page_config(page_title="Audio Recorder Demo", layout="wide")
st.title("Streamlit Audio Recorder")

# Session state
if "audio_chunks" not in st.session_state:
    st.session_state.audio_chunks = []
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None


class AudioRecorder:
    def __init__(self):
        self.audio_chunks = []

    def recv(self, frame):
        """Receive and process audio frames"""
        try:
            if st.session_state.recording:
                # Convert audio frame to numpy array
                audio_data = frame.to_ndarray()
                self.audio_chunks.append(audio_data.copy())

                # Also store in session state for persistence
                st.session_state.audio_chunks.append(audio_data.copy())

                # Debug info
                if len(self.audio_chunks) % 50 == 0:  # Log every 50 frames
                    print(f"Received {len(self.audio_chunks)} audio frames")

            return frame
        except Exception as e:
            print(f"Error in recv: {e}")
            return frame

    def get_audio_data(self):
        """Combine all audio chunks into single array"""
        if not self.audio_chunks:
            return None

        # Concatenate all audio frames
        combined = np.concatenate(self.audio_chunks, axis=1)
        return combined


# WebRTC component
webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,  # Buffer size
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "audio": {
            "sampleRate": 44100,
            "channelCount": 1,
            "echoCancellation": True,
            "noiseSuppression": True,
        },
        "video": False
    },
    audio_processor_factory=AudioRecorder,
)

# Control buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎤 Start Recording"):
        st.session_state.recording = True
        st.session_state.audio_chunks = []
        st.success("Recording started! Speak into your microphone.")
        st.rerun()

with col2:
    if st.button("⏹️ Stop Recording"):
        st.session_state.recording = False
        st.success("Recording stopped!")

        # Process audio from session state
        if st.session_state.audio_chunks:
            try:
                # Combine all audio chunks
                all_frames = np.concatenate(st.session_state.audio_chunks, axis=1)

                # Convert to bytes
                audio_bytes = all_frames.tobytes()

                # Create AudioSegment
                audio_segment = AudioSegment(
                    data=audio_bytes,
                    sample_width=2,  # 16-bit
                    frame_rate=44100,
                    channels=1
                )

                # Export to MP3
                buffer = io.BytesIO()
                audio_segment.export(buffer, format="mp3", bitrate="128k")
                st.session_state.audio_bytes = buffer.getvalue()

                # Save to file
                filename = f"recording_{int(time.time())}.mp3"
                with open(filename, "wb") as f:
                    f.write(st.session_state.audio_bytes)

                st.success(f"Audio saved as {filename}")

            except Exception as e:
                st.error(f"Error processing audio: {e}")
        else:
            st.warning("No audio data was recorded!")
        st.rerun()

with col3:
    if st.button("🗑️ Clear"):
        st.session_state.audio_chunks = []
        st.session_state.audio_bytes = None
        st.session_state.recording = False
        st.success("Recording cleared!")
        st.rerun()

# Status information
st.subheader("📊 Status")
st.write(f"**Recording:** {st.session_state.recording}")
st.write(f"**Audio chunks collected:** {len(st.session_state.audio_chunks)}")
st.write(f"**WebRTC state:** {webrtc_ctx.state if webrtc_ctx else 'Not initialized'}")

if webrtc_ctx and webrtc_ctx.audio_processor:
    st.write(f"**Processor frames:** {len(webrtc_ctx.audio_processor.audio_chunks)}")

# Audio playback
if st.session_state.audio_bytes:
    st.subheader("🎵 Recorded Audio")
    st.audio(st.session_state.audio_bytes, format="audio/mp3")

    # Download button
    st.download_button(
        label="📥 Download MP3",
        data=st.session_state.audio_bytes,
        file_name=f"recording_{int(time.time())}.mp3",
        mime="audio/mp3"
    )

# Debug information
with st.expander("🔍 Debug Info"):
    st.write("This helps identify what's happening:")

    if webrtc_ctx:
        st.write("**WebRTC Context:**", webrtc_ctx.state)
        if webrtc_ctx.audio_processor:
            st.write("**Audio Processor:**", f"{len(webrtc_ctx.audio_processor.audio_chunks)} frames")
        else:
            st.write("**Audio Processor:** Not available")

    st.write("**Session State:**")
    st.json({
        "recording": st.session_state.recording,
        "audio_chunks_count": len(st.session_state.audio_chunks),
        "has_audio_bytes": st.session_state.audio_bytes is not None
    })

# Troubleshooting tips
with st.expander("❓ Troubleshooting Tips"):
    st.markdown("""
    1. **Check browser permissions** - Allow microphone access when prompted
    2. **Use HTTPS** - streamlit-webrtc often requires HTTPS for microphone access
    3. **Try different browsers** - Chrome usually works best
    4. **Check your microphone** - Make sure it's not muted and is the default device
    5. **Run Streamlit with** `streamlit run your_script.py --server.enableCORS=false`
    """)