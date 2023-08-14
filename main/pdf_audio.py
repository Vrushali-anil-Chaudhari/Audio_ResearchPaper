
from dotenv import load_dotenv
from pytesseract import image_to_string
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
import streamlit as st
import multiprocessing
from tempfile import NamedTemporaryFile
import pandas as pd
import json
import requests

load_dotenv()

# pdf to image
def convert_pdf_to_images(file_path, scale=300/72):
    pdf_file = pdfium.PdfDocument(file_path)
    page_indices = [i for i in range(len(pdf_file))]
    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )
    final_images = []
    for i, image in zip(page_indices, renderer):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))
    
    return final_images

# 2. Extract text from images via pytesseract


def extract_text_from_img(list_dict_final_images):
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []
    for index, image_bytes in enumerate(image_list):
        image = Image.open(BytesIO(image_bytes))
        print(image)
        raw_text = str(image_to_string(image))
        image_content.append(raw_text)

    return "\n".join(image_content)


def extract_content_from_url(url: str):
    images_list = convert_pdf_to_images(url)
    text_with_pytesseract = extract_text_from_img(images_list)
    return text_with_pytesseract


# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from pydub import AudioSegment

import scipy

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = BarkModel.from_pretrained("suno/bark")
# model = model.to(device)
# processor = AutoProcessor.from_pretrained("suno/bark")

if __name__ == '__main__':
    images_list = convert_pdf_to_images(r"Audio_ResearchPaper\data\sample.pdf")
    text_with_pytesseract = extract_text_from_img(images_list)
    t_all = text_with_pytesseract
    text = t_all.split(" ")
    start = 0 
    end  = 70
    j = 1
    while(end <= len(text)):
        t = ""
        for i  in range(start,end):
            t = t + text[i] + " "
            
        start = end
        end = end + 70
        if end >= len(text):
            end = len(text)
     
        inputs = processor(text=t, return_tensors="pt")

        # load xvector containing speaker's voice characteristics from a dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        filename = r"result" +str(j)+ ".wav"
        j +=1
        sf.write(filename, speech.numpy(), samplerate=16000)

    # Replace these paths with the actual paths to your WAV files
    input_wav_files = []
    for i in range(1,j):
         input_wav_files.append(r"result\speech" +str(i)+".wav")
    # Load the first WAV file to initialize the combined audio
    combined_audio = AudioSegment.from_wav(input_wav_files[0])

    # Loop through the remaining WAV files and concatenate them to the combined audio
    for wav_file in input_wav_files[1:]:
        audio = AudioSegment.from_wav(wav_file)
        combined_audio += audio

    # Specify the path for the output merged WAV file
    output_wav_file = r"Audio_ResearchPaper\final_result\merged_output.wav"

    # Export the combined audio as a single WAV file
    combined_audio.export(output_wav_file, format="wav")

    print("Merged WAV file saved as", output_wav_file)

