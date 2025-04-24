import streamlit as st
import os
import config
from model_train import ModelTrain


class UI:
    def __init__(self):
        if "training" not in st.session_state:
            st.session_state.training = False

        if "model_train" not in st.session_state:
            st.session_state.model_train = None

        if "progress_bar" not in st.session_state:
            st.session_state.progress_bar = None

    def build(self):
        st.set_page_config(page_title="Picture Categorizer CNN", page_icon="🖼️")
        st.title("Your image identifier 🖼️")
        st.markdown("`['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']`")
        upload = None

        self.__build_sidebar()

        if st.session_state.model_train is not None:
            upload = st.file_uploader(
                label="Send Images",
                type=["png", "jpg"],
                accept_multiple_files=False,
            )
        else:
            st.markdown("#### Train or load a model to continue...")

        if upload is not None:
            st.image(upload)
            result_identification = st.session_state.model_train.evaluate(upload)
            st.markdown(f"##### This is a: `{result_identification}`")

    def __build_sidebar(self):
        with st.sidebar.container():
            st.markdown("# Model config")
            st.text("Train your model")
            epochs = st.text_input("Epochs:", placeholder="5")
            st.button(
                "Train",
                on_click=self.__on_train,
                disabled=epochs is None or epochs == "" or st.session_state.training,
                args=(epochs, False),
            )

            st.markdown("---").empty()
            st.markdown("---")

            st.text("Load a trained model")
            st.button("Load", on_click=self.__on_load, disabled=self.__disable_load())

            if st.session_state.model_train is not None:
                epochs = st.text_input("Epochs:", placeholder="5", key="epoch-load")
                st.button("Continue train", on_click=self.__on_train, args=(epochs, True))

            st.markdown("---").empty()
            st.markdown("---")

            st.text("Save your model trained")
            st.button("Save", on_click=self.__save_model, disabled=st.session_state.model_train is None)

            st.markdown("---").empty()
            st.markdown("---")

            if st.session_state.progress_bar is None:
                st.session_state.progress_bar = st.progress(0).empty()

    def __on_train(self, epochs: str, load=False):
        st.session_state.training = True
        epochs = int(epochs)

        st.session_state.progress_bar.progress(0, "Load datasets...")
        st.session_state.model_train = ModelTrain(load)

        st.session_state.progress_bar.progress(0, "Load model...")
        train_result = st.session_state.model_train.train(epochs)

        for result, loss_value in train_result:
            progress_status = (100 / epochs) * result / 100
            st.session_state.progress_bar.progress(progress_status, f"Train {result} of {epochs} | Loss: {loss_value}...")

        st.session_state.training = False

    def __on_load(self):
        if not self.__disable_load():
            st.session_state.progress_bar.progress(50, "Loading...")
            st.session_state.model_train = ModelTrain(True)
            st.session_state.progress_bar.empty()

    def __save_model(self):
        st.session_state.model_train.save_model()

    def __disable_load(self):
        return not os.path.exists(config.TRAIN_DATA_FULL_PATH)
