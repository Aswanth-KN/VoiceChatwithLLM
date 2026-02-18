import requests
import os
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')

# Set HF_TOKEN as environment variable for transformers library
if HUGGINGFACE_API_KEY:
    os.environ['HF_TOKEN'] = HUGGINGFACE_API_KEY
    os.environ['HUGGINGFACE_HUB_TOKEN'] = HUGGINGFACE_API_KEY

BASE_URL = "https://router.huggingface.co/hf-inference/models"

# Global model cache to avoid reloading models on every request
_models = {
    'whisper': None,
    'whisper_processor': None,
    'llm': None,
    'llm_tokenizer': None,
    'tts': None,
    'tts_processor': None,
    'tts_vocoder': None,
    'speaker_embeddings': None,
    'device': "cuda" if torch.cuda.is_available() else "cpu"
}


def speech_to_text(audio_binary):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import soundfile as sf
    import subprocess
    import tempfile

    print('Audio size:', len(audio_binary))

    try:
        # Load models once and cache them
        if _models['whisper'] is None:
            print("Loading Whisper model...")
            _models['whisper_processor'] = WhisperProcessor.from_pretrained("openai/whisper-base")
            _models['whisper'] = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            _models['whisper'] = _models['whisper'].to(_models['device'])
        
        processor = _models['whisper_processor']
        model = _models['whisper']

        # Save audio to temporary file and convert to WAV using ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_input:
            temp_input.write(audio_binary)
            temp_input_path = temp_input.name
        
        temp_output_path = tempfile.mktemp(suffix='.wav')
        
        try:
            # Convert to WAV format using ffmpeg
            result = subprocess.run([
                'ffmpeg', '-y', '-i', temp_input_path, 
                '-ar', '16000',  # Whisper expects 16kHz
                '-ac', '1',      # Mono
                '-f', 'wav',
                temp_output_path
            ], check=True, capture_output=True, text=True)
            
            # Read the converted audio
            audio_data, sample_rate = sf.read(temp_output_path)
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
        
        # Process the audio
        input_features = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_features
        input_features = input_features.to(_models['device'])

        # Generate transcription with faster settings
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_new_tokens=128)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        print("Transcription:", transcription)
        return transcription

    except Exception as e:
        print(f"STT exception: {e}")
        import traceback
        traceback.print_exc()
        return "Error processing audio"


def text_to_speech(text, voice=""):
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    import soundfile as sf
    import io

    try:
        # Load models once and cache them
        if _models['tts'] is None:
            print("Loading TTS model...")
            _models['tts_processor'] = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            _models['tts'] = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            _models['tts_vocoder'] = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            _models['tts'] = _models['tts'].to(_models['device'])
            _models['tts_vocoder'] = _models['tts_vocoder'].to(_models['device'])
            
            # Create and cache speaker embeddings
            torch.manual_seed(42)
            _models['speaker_embeddings'] = torch.randn((1, 512)) * 0.5
            _models['speaker_embeddings'] = _models['speaker_embeddings'].to(_models['device'])
        
        processor = _models['tts_processor']
        model = _models['tts']
        vocoder = _models['tts_vocoder']
        speaker_embeddings = _models['speaker_embeddings']

        inputs = processor(text=text.strip(), return_tensors="pt")
        inputs = {k: v.to(_models['device']) for k, v in inputs.items()}

        # Generate speech with cached models
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

        # Convert tensor to numpy array
        speech_numpy = speech.cpu().numpy()
        
        # Convert to bytes using soundfile
        buffer = io.BytesIO()
        sf.write(buffer, speech_numpy, samplerate=16000, format='WAV')
        audio_bytes = buffer.getvalue()
        
        return audio_bytes
    except Exception as e:
        print(f'TTS Error: {e}')
        import traceback
        traceback.print_exc()
        return b''

def openai_process_message(user_message):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    try:
        # Load models once and cache them
        if _models['llm'] is None:
            print("Loading LLM model...")
            model_name = "google/flan-t5-base"
            _models['llm_tokenizer'] = AutoTokenizer.from_pretrained(model_name)
            _models['llm'] = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if _models['device'] == "cuda" else torch.float32,
            )
            _models['llm'] = _models['llm'].to(_models['device'])
        
        tokenizer = _models['llm_tokenizer']
        model = _models['llm']

        # Better prompt formatting for conversational responses
        input_text = f"You are a helpful assistant. User: {user_message}\nAssistant:"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(_models['device'])

        # Generate response with faster settings (greedy decoding, no sampling)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,  # Reduced from 150
                do_sample=False,     # Greedy decoding for speed
                early_stopping=True
            )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("LLM response:", response[:200])
        return response

    except Exception as e:
        print(f"LLM exception: {e}")
        return f"Error processing message: {str(e)}"
