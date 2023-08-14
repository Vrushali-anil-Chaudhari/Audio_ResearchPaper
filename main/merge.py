from pydub import AudioSegment

input_wav_files = []
for i in range(1,4680):
    input_wav_files.append(r"result\speech" +str(i)+".wav")
    # Load the first WAV file to initialize the combined audio
combined_audio = AudioSegment.from_wav(input_wav_files[0])

# Loop through the remaining WAV files and concatenate them to the combined audio
for wav_file in input_wav_files[1:]:
    audio = AudioSegment.from_wav(wav_file)
    combined_audio += audio

# Specify the path for the output merged WAV file
output_wav_file = r"C:\Users\vrush\OneDrive\Documents\Desktop\demoo\didi\final_result\merged_output.wav"

# Export the combined audio as a single WAV file
combined_audio.export(output_wav_file, format="wav")

print("Merged WAV file saved as", output_wav_file)

