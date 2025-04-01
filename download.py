# Pull all the models used

def bert():
    from transformers import BertConfig, BertForSequenceClassification, BertForTokenClassification, BertModel, BertTokenizer
    model_names = [
        "prajjwal1/bert-tiny",
        "bert-base-uncased",
        "bert-large-uncased",
        "textattack/bert-base-uncased-SST-2",
        "assemblyai/bert-large-uncased-sst2",
        "yanekyuk/bert-uncased-keyword-extractor"
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        model = BertForSequenceClassification.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        print(f"Downloaded {model_name} successfully.")
        del model, tokenizer

def deit():
    from transformers import ViTForImageClassification, AutoFeatureExtractor
    model_names = [
        "facebook/deit-base-patch16-224",
        "facebook/deit-small-patch16-224"
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        model = ViTForImageClassification.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        print(f"Downloaded {model_name} successfully.")
        del model, feature_extractor

def falcon():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_names = [
        "tiiuae/falcon-7b",
        "tiiuae/falcon-7b-instruct"
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Downloaded {model_name} successfully.")
        del model, tokenizer

def flant5():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config, pipeline
    model_names = [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large"
    ]
    for model_name in model_names:
        print(f"Downloading {model_name}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = T5Config.from_pretrained(model_name)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        print(f"Downloaded {model_name} successfully.")
        del model, tokenizer, config, pipe

def hrnet():
    from torchvision import transforms
    from datasets import load_dataset
    from transformers import AutoFeatureExtractor
    from pytorchcv.model_provider import get_model as ptcv_get_model

    model_names = [
        "hrnet_w18_small_v2",
        "hrnetv2_w18",
        "hrnetv2_w30",
        "hrnetv2_w32",
        "hrnetv2_w40",
        "hrnetv2_w44",
        "hrnetv2_w48",
        "hrnetv2_w64"
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        model = ptcv_get_model(model_name, pretrained=True)
        print(f"Downloaded {model_name} successfully.")
        del model#, feature_extractor

def inception_v4():
    #timm
    import timm
    from datasets import load_dataset
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torch.utils.data import DataLoader

    model_names = [
        "inception_v4"
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        model = timm.create_model(model_name, pretrained=True)
        del model

def mobilenet_v1():
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoImageProcessor

    model_names = [
        "google/mobilenet_v1_0.75_192",
        "google/mobilenet_v1_1.0_224",
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        print(f"Downloaded {model_name} successfully.")
        del model, feature_extractor


def mobilenet_v2():
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification

    model_names = [
        "google/mobilenet_v2_1.0_224",
        "google/mobilenet_v2_0.75_160",
        "google/mobilenet_v2_0.35_96"
    ]
    for model_name in model_names:
        print(f"Downloading {model_name}...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        print(f"Downloaded {model_name} successfully.")
        del model, feature_extractor

def mobilenet_v3():
    import timm
    from datasets import load_dataset
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torch.utils.data import DataLoader

    model_names = [
        "hf_hub:timm/mobilenetv3_small_100.lamb_in1k",
        "hf_hub:timm/mobilenetv3_large_100.ra_in1k"
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        model = timm.create_model(model_name, pretrained=True)
        config_model = resolve_data_config({}, model=model)
        transform = create_transform(**config_model)
        print(f"Downloaded {model_name} successfully.")
        del model

def resnet():
    from torchvision import models
    from datasets import load_dataset
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification

    model_names = [
        "microsoft/resnet-18",
        "microsoft/resnet-50",
        "microsoft/resnet-101"
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        print(f"Downloaded {model_name} successfully.")
        del model, feature_extractor


def stable_diffusion():
    from diffusers import StableDiffusionPipeline
    from torch.utils.data import DataLoader
    from torchmetrics.multimodal.clip_score import CLIPScore
    from torchvision.transforms.functional import pil_to_tensor
    model_names = [
        "CompVis/stable-diffusion-v1-4"
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        model = StableDiffusionPipeline.from_pretrained(model_name)
        print(f"Downloaded {model_name} successfully.")
        del model

def t5():
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    model_names = [
        "t5-small",
        "t5-base",
        "t5-large"
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        print(f"Downloaded {model_name} successfully.")
        del model, tokenizer

def vit():
    from transformers import ViTForImageClassification, AutoFeatureExtractor
    model_names = [
        "google/vit-base-patch16-224",
        "google/vit-large-patch16-224"
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        model = ViTForImageClassification.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        print(f"Downloaded {model_name} successfully.")
        del model, feature_extractor

def unet():
    import torch
    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
    )


def vovnet_v1():
    import torch
    from datasets import load_dataset
    from pytorchcv.model_provider import get_model as ptcv_get_model
    from torch.utils.data import DataLoader
    from torchvision import transforms

    model_names = [
        "vovnet27s",
        "vovnet39",
        "vovnet57"
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        model = ptcv_get_model(model_name, pretrained=True)
        print(f"Downloaded {model_name} successfully.")
        del model

def vovnet_v2():
    import torch
    from datasets import load_dataset
    from pytorchcv.model_provider import get_model as ptcv_get_model
    from torch.utils.data import DataLoader
    from torchvision import transforms

    import timm
    from datasets import load_dataset
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torch.utils.data import DataLoader

    model_names = [
        "ese_vovnet19b_dw",
        "ese_vovnet39b",
        # "ese_vovnet99b"
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        model = timm.create_model(model_name, pretrained=True)
        config_model = resolve_data_config({}, model=model)
        transform = create_transform(**config_model)
        print(f"Downloaded {model_name} successfully.")
        del model
    

def whisper():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    model_names = [
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v2",
    ]

    for model_name in model_names:
        print(f"Downloading {model_name}...")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        processor = WhisperProcessor.from_pretrained(model_name)
        print(f"Downloaded {model_name} successfully.")
        del model, processor

def yolov5():
    import torch
    model = model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    print("Downloaded yolov5 successfully.")
    del model

if __name__ == "__main__":
    # Uncomment the models you want to download
    # bert()
    # deit()
    # falcon()
    # flant5()
    hrnet()
    # inception_v4()
    # mobilenet_v1()
    # mobilenet_v2()
    # mobilenet_v3()
    # resnet()
    #stable_diffusion()
    #t5()
    #vit()
    #unet()
    #vovnet_v1()
    # vovnet_v2()
    # whisper()
    # yolov5()  # This is the only one that doesn't require any additional libraries