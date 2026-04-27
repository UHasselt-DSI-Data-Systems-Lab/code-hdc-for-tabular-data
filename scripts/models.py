import torchhd
from gensim.models.fasttext import load_facebook_model
from scripts.utils import load_config

cfg = load_config()


class PreTrainedModel:
    def __init__(self, name: str, model_path: str):
        self.name = name
        self.model = load_facebook_model(model_path)
        self.vector_size = self.model.vector_size

    def get_vector(self, word: str):
        if self.name == cfg["models"]["fasttext"]["name"]:
            return torchhd.HRRTensor(self.model.wv[word].copy())
