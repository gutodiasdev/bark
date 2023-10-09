# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from IPython.display import Audio
from scipy.io.wavfile import write as write_wav

from bark.api import generate_audio
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic



class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        semantic_path = "semantic_output/pytorch_model.bin" # set to None if you don't want to use finetuned semantic
        coarse_path = "coarse_output/pytorch_model.bin" # set to None if you don't want to use finetuned coarse
        fine_path = "fine_output/pytorch_model.bin" # set to None if you don't want to use finetuned fine
        use_rvc = True # Set to False to use bark without RVC
        rvc_name = 'mi-test'
        rvc_path = f"Retrieval-based-Voice-Conversion-WebUI/weights/{rvc_name}.pth"
        index_path = f"Retrieval-based-Voice-Conversion-WebUI/logs/{rvc_name}/added_IVF256_Flat_nprobe_1_{rvc_name}_v2.index"
        device="cuda:0"
        is_half=True
        preload_models(
            text_use_gpu=True,
            text_use_small=False,
            text_model_path=semantic_path,
            coarse_use_gpu=True,
            coarse_use_small=False,
            coarse_model_path=coarse_path,
            fine_use_gpu=True,
            fine_use_small=False,
            fine_model_path=fine_path,
            codec_use_gpu=True,
            force_reload=False,
            path="models"
        )

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=1.5
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        text_prompt = "Olá, meu nome é Augusto. Este é o clone da minha voz"
        voice_name = "cloned_voice" # use your custom voice name here if you have on

        filepath = "output/audio_cloned_augusto.wav"
        audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)
        write_wav(filepath, SAMPLE_RATE, audio_array)
