import os
import torch
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence
from torchvision import transforms
from torch.utils.data import DataLoader
from spleeter.separator import Separator
from models.dinet import DINet
from datasets.video_dataset import VideoDataset
from utils.inference_utils import boost_audio, detect_silence_pydub

# Assuming the required packages are already imported

def separate_speech_and_noise(audio_path):
    """
    Separates speech from background noise or music using Spleeter.
    This function will separate the input audio into vocals and accompaniment.

    Args:
    - audio_path: Path to the audio file

    Returns:
    - speech_audio: Tensor of speech audio (isolated)
    - noise_audio: Tensor of background noise or music (isolated)
    """
    # Initialize Spleeter with the 2stems model (vocals and accompaniment)
    separator = Separator('spleeter:2stems')  # Using the 2 stems model for vocals + accompaniment separation

    # Perform the separation
    separator.separate(audio_path)

    # Load the separated tracks (vocals and accompaniment)
    vocals_path = audio_path.replace('.wav', '/vocals.wav')
    accompaniment_path = audio_path.replace('.wav', '/accompaniment.wav')

    # Load the separated audio files
    speech_waveform, sample_rate = torchaudio.load(vocals_path)
    noise_waveform, _ = torchaudio.load(accompaniment_path)

    return speech_waveform, noise_waveform

# Initialize model and options
model = DINet().to('cuda')
model.eval()

# Set paths and configurations
opt = {
    'driving_audio_path': 'path_to_audio.wav',
    'video_frames_path': 'path_to_video_frames',
    'output_video_path': 'output_path.mp4',
    'fps': 25
}

# Boost audio and detect silences
print('Boosting and detecting silence in driving audio')
boosted_audio = boost_audio(opt['driving_audio_path'])
absolute_silences = detect_silence_pydub(boosted_audio)

# Separate speech and background noise/music using Spleeter
speech_audio, noise_audio = separate_speech_and_noise(opt['driving_audio_path'])

# Data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset and DataLoader
video_dataset = VideoDataset(opt['video_frames_path'], transform=transform)
data_loader = DataLoader(video_dataset, batch_size=1, shuffle=False)

# Processing frames
for clip_idx, (crop_frame_tensor, ref_img_tensor, deepspeech_tensor) in enumerate(data_loader):
    crop_frame_tensor = crop_frame_tensor.to('cuda')
    ref_img_tensor = ref_img_tensor.to('cuda')
    deepspeech_tensor = deepspeech_tensor.to('cuda')

    # Determine timestamp for silence detection
    clip_end_index = clip_idx + 1
    timestamp_end = clip_end_index / opt['fps']

    # Check if the current frame falls within a silence segment
    is_silence = False
    for segment in absolute_silences:
        if segment['start'] <= timestamp_end <= segment['end']:
            is_silence = True
            break

    # Prepare tensor for silent frames
    silent_speech_tensor = torch.from_numpy(
        np.zeros_like(deepspeech_tensor.cpu().numpy())
    ).permute(1, 0).unsqueeze(0).float().to('cuda')

    # Forward pass through the model
    with torch.no_grad():
        pre_frame = model(
            crop_frame_tensor, 
            ref_img_tensor, 
            silent_speech_tensor if is_silence else deepspeech_tensor
        )

    # Save or process pre_frame as needed
    # For example, save it as a frame in the output video
    # Your saving logic goes here

print("Inference completed!")
