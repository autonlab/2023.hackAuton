# Ref https://github.com/ezzcodeezzlife/dalle2-in-python
# Ref https://towardsdatascience.com/speech-to-text-with-openais-whisper-53d5cea9005e
# Ref https://python.plainenglish.io/creating-an-awesome-web-app-with-python-and-streamlit-728fe100cf7
import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import List
import whisper
import av
import numpy as np
import pydub
import streamlit as st
from tqdm import tqdm
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from dalle2 import Dalle2
from PIL import Image

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

# Initialize the OpenAI API with your API key
import openai
openai.api_key = ''

prompt = """I am a doctor, I would like you to check my prescription:
medical history: Hypertension, Type 2 Diabetes, and Asthma.
symptoms: Persistent cough, fever, and fatigue.
My prescription: Lisinopril 10mg daily, Metformin 500mg twice daily, and Albuterol as needed for asthma attacks.
Drug contexts:
- Lisinopril: Ingredients: ACE inhibitor. Adverse effects: Dizziness, dry cough, elevated blood potassium levels.
- Metformin: Ingredients: Oral antihyperglycemic agent. Adverse effects: Stomach upset, diarrhea, low blood sugar.
- Albuterol: Ingredients: Bronchodilator. Adverse effects: Tremors, nervousness, increased heart rate.

Please answer the following questions in concise point form, taking into account the provided drug context:
- Possible interactions between prescribed drugs?
- Adverse effect of given drugs that are specifically related to patient’s pre-existing conditions and medical history?

At the end of your answer, evaluate the level of dangerousness of this treatment, based on interactions and adverse effects. Dangerousness is categorized as: LOW, MEDIUM, HIGH
Your answer should look like this:
`
* interactions:
- <interaction 1>
- <interaction 2>
- ...

* adverse effects:
- <adverse effect 1>
- <adverse effect 2>
- ...`

* dangerousness: <LOW / MEDIUM / HIGH>

Note that you don't have to include any interactions or adverse effect, only those that are necessary.
"""
def get_drug_info_string(drug_names):
    # Make the drug_to_info dictionary into a string with each line of the form drug: info
    drug_info_string = ""
    for drug in drug_names:
        info = search_openfda_drug(drug)
        drug_info_string += drug + ": " + str(trim_openfda_response(search_openfda_drug(drug))) + "\r\n"
    return drug_info_string
import requests

def trim_openfda_response(json_response):
    """Trim the openFDA JSON response to include only specific fields.

    Parameters:
    - json_response (dict): The raw JSON response from the openFDA API.

    Returns:
    - dict: A trimmed version of the JSON response.
    """

    # List of desired fields
    desired_fields = [
        "spl_product_data_elements",
        "boxed_warning",
        "contraindications",
        "drug_interactions",
        "adverse_reactions",
        "warnings"
    ]

    trimmed_response = {}

    # Check if results are present in the response
    if 'results' in json_response:
        for field in desired_fields:
            if field in json_response['results'][0]:
                trimmed_response[field] = json_response['results'][0][field]

    return trimmed_response

def search_openfda_drug(drug_name):
    """Search for a drug in the openFDA database.

    Parameters:
    - drug_name (str): The name of the drug to search for.

    Returns:
    - dict: The JSON response from the openFDA API containing drug information, or None if there's an error.
    """

    base_url = "https://api.fda.gov/drug/label.json"
    query = f"?search=openfda.generic_name:{drug_name}&limit=1"

    try:
        response = requests.get(base_url + query)

        # Check for successful request
        if response.status_code == 200:
            return response.json()

    except requests.RequestException:
        # If any request-related exception occurs, simply return None
        print(f"Error encountered searching for drug {drug_name} with code {response.status_code}.")

    return None

def ask_gpt(question, model="gpt-3.5-turbo"):
    """
    Query the GPT-3.5 Turbo model with a given question.

    Parameters:
    - question (str): The input question or prompt for the model.

    Returns:
    - str: The model's response.
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a knowledgeable medical database designed to provide concise and direct answers to medical questions."},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message['content']

def parse_gpt(question):
    """
    Query the GPT-3.5 Turbo model with a given question.

    Parameters:
    - question (str): The input question or prompt for the model.

    Returns:
    - str: The model's response.
    """

    # 1 parse text to replace critical information in the prompt
    # 2 send parsed text in OpenAPI

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a knowledgeable medical database designed to provide concise and direct answers to medical questions."},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message['content']

# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def main():
    st.header("openFDA Medical Records Evaluation")
    st.markdown(
        """
This demo app is using [DeepSpeech](https://github.com/mozilla/DeepSpeech),
an open speech-to-text engine.

A pre-trained model released with
[v0.9.3](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3),
trained on American English is being served.
"""
    )

    # https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3
    MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
    LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
    MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.pbmm"
    LANG_MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.scorer"

    #download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=188915987)
    #download_file(LANG_MODEL_URL, LANG_MODEL_LOCAL_PATH, expected_size=953363776)

    lm_alpha = 0.931289039105002
    lm_beta = 1.1834137581510284
    beam = 100

    medical_text_page = "Medical text evaluation" # summarize notes and identify risk from notes (useful for change in doctors)
    voice_to_text_page = "Voice to medical text"  # use voice to text and identify risk, could be use in case
    image_to_text_page = "Image to medical text"  # use image to text and identify risk, could be use in case
    all_in_one_page = "All modalities"            # use all modalities
    sound_only_page = "Sound only (sendonly)"
    with_video_page = "With video (sendrecv)"
    text_only_page = "Text only for DALLE2"
    app_mode = st.selectbox(
        "Choose the app mode", 
        # [sound_only_page, with_video_page, text_only_page, medical_text_page]
        [medical_text_page, voice_to_text_page, image_to_text_page, all_in_one_page]
    )



    if app_mode == sound_only_page:
        app_sst(
            str(MODEL_LOCAL_PATH), str(LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
        )
    elif app_mode == with_video_page:
        app_sst_with_video(
            str(MODEL_LOCAL_PATH), str(LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
        )
    elif app_mode == medical_text_page:
        form = st.form(key='my-form')
        text = form.text_input('Medical text description')
        submit = form.form_submit_button('Submit')

        st.write('Press submit to evaluate medical notes')

        if submit:
            # res = parse_gpt(text + "Organize the answers in 3 parts, first is pre-existing conditions, second is symptoms, third is prescriptions. Sample output for drugs should be the end of the answer as DRUG_NAMES: <drug 1>, <drug 2>, <drug 3>...")
            parsed_notes = ask_gpt(f"""
                    Please parse the following medical note in point form, without losing any important information:
                    `{text}`

                    your answer should look like: 
                    `Patient's medical history:
                    - <point 1>
                    - <point 2>
                    - ...

                    Patient's symptoms:
                    - <point 1>
                    - <point 2>
                    - ...

                    Prescription:
                    - ...

                    DRUGS: <drug 1>, <drug 2>, ...
                    `
                    Please be reminded to give the generic names for the drugs
                    """)
            st.write(parsed_notes)
            # Extract the drugs portion from the notes
            drug_line = [line for line in parsed_notes.split("\n") if line.startswith("DRUGS:")][0]

            # Strip the "DRUGS: " prefix and split the drugs by ", "
            drugs = drug_line.replace("DRUGS: ", "").strip().split(", ")
            
            # Go to FDA
            drug_info_string = get_drug_info_string(drugs)
            # st.write(drug_info_string)

            #  #
            risk = ask_gpt(f"""I am a doctor, I would like you to check my prescription:
                {parsed_notes}

                Drug contexts:
                {drug_info_string}

                Please answer the following questions in concise point form, taking into account the provided drug context:
                - Possible interactions between prescribed drugs?
                - Adverse effect of given drugs, only answer those that are specifically related to patient’s pre-existing conditions and symptoms?

                At the end of your answer, evaluate the level of dangerousness of this treatment, based on interactions and adverse effects that are specific to the patient. Dangerousness is categorized as: LOW, MEDIUM, HIGH
                Your answer should look like this (you should include the * where specified):
                `
                * INTERACTIONS:
                - <interaction 1>
                - <interaction 2>
                - ...

                * ADVERSE EFFECTS:
                - <adverse effect 1>
                - <adverse effect 2>
                - ...`

                * DANGEROUSNESS: <LOW / MEDIUM / HIGH>

                Note that you don't have to include any interactions or adverse effect, only those that are necessary.
                """, model = 'gpt-3.5-turbo-16k')
            # st.write(res)
            st.write(risk)

    elif app_mode == text_only_page:
        form = st.form(key='my-form')
        text = form.text_input('Image description')
        submit = form.form_submit_button('Submit')

        st.write('Press submit to generate image')

        if submit:
            app_sst_dalle2(text)

        # form = st.form(key='my_form')
        # text = form.text_input(label='Image Description')
        # submit_button = form.form_submit_button(label='Submit')
        # if submit_button:
        #     app_sst_dalle2(form.text)

        #text = st.text_input('Image description')
        #if st.form_submit_button('Generate') == True:
        #    app_sst_dalle2(text)


def app_sst_dalle2(text):
    dalle = Dalle2("sess-TotC46rSs5pbqdXTRy75cr81ynLJALwa2b3rdxeh")
    #generations = dalle.generate(text)
    file_paths = dalle.generate_and_download(text)
    print(file_paths)
    #generations = dalle.generate_amount(text, 8) # Every generation has batch size 4 -> amount % 4 == 0 works best
    for file in file_paths:
        image = Image.open(file)
        st.image(image, caption=text)

def app_sst(model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int):
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None

    while True:
        if webrtc_ctx.audio_receiver:
            if stream is None:
                from deepspeech import Model
                # https://github.com/openai/whisper
                # model = whisper.load_model(“large”)

                model = Model(model_path)
                model.enableExternalScorer(lm_path)
                model.setScorerAlphaBeta(lm_alpha, lm_beta)
                model.setBeamWidth(beam)

                stream = model.createStream()

                status_indicator.write("Model loaded.")

            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                    model.sampleRate()
                )
                buffer = np.array(sound_chunk.get_array_of_samples())
                stream.feedAudioContent(buffer)
                text = stream.intermediateDecode()
                text_output.markdown(f"**Text:** {text}")
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break


def app_sst_with_video(
    model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int
):
    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])

    async def queued_audio_frames_callback(
        frames: List[av.AudioFrame],
    ) -> av.AudioFrame:
        with frames_deque_lock:
            frames_deque.extend(frames)

        # Return empty frames to be silent.
        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)

        return new_frames

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text-w-video",
        mode=WebRtcMode.SENDRECV,
        queued_audio_frames_callback=queued_audio_frames_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None

    while True:
        if webrtc_ctx.state.playing:
            if stream is None:
                from deepspeech import Model

                model = Model(model_path)
                model.enableExternalScorer(lm_path)
                model.setScorerAlphaBeta(lm_alpha, lm_beta)
                model.setBeamWidth(beam)

                stream = model.createStream()

                status_indicator.write("Model loaded.")

            sound_chunk = pydub.AudioSegment.empty()

            audio_frames = []
            with frames_deque_lock:
                while len(frames_deque) > 0:
                    frame = frames_deque.popleft()
                    audio_frames.append(frame)

            if len(audio_frames) == 0:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                    model.sampleRate()
                )
                buffer = np.array(sound_chunk.get_array_of_samples())
                stream.feedAudioContent(buffer)
                text = stream.intermediateDecode()
                text_output.markdown(f"**Text:** {text}")
        else:
            status_indicator.write("Stopped.")
            break

# a raccoon astronaut with the cosmos reflecting on the glass of his helmet dreaming of the stars
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.redd.it/zung2u9zryb91.png");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

#add_bg_from_url() 


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
