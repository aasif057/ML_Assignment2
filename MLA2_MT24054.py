import random
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
from PIL import Image
import sys
from scipy.io import wavfile
sys.stdout.flush()

# Beginner

# Q1

#1.a    Code for 1.a

def preprocess_image(image_path):
    # Load image using PIL
    img = Image.open(image_path)
    # Convert to grayscale
    img_gray = img.convert("L")
    # save data from image
    img_color_array = list(img.getdata())
    img_gray_array = list(img_gray.getdata())
    img_color_array = [img_color_array[i:i + img.width] for i in range(0, len(img_color_array), img.width)]
    img_gray_array = [img_gray_array[i:i + img_gray.width] for i in range(0, len(img_gray_array), img_gray.width)]
    return img_color_array, img_gray_array    

# function to normalize the image data
def normalize_img(data):
  return [[(pixel - 127.5) / 127.5 for pixel in row] for row in data]

#function to plot the color vs grayscale image, unnormalized vs normalized image.
def plot_image_data(img_color1, img_gray1):
    norm_gray1 = normalize_img(img_gray1)
    # Create a subplot for side-by-side images
    flattened_gray = [pixel for row in img_gray1 for pixel in row]
    flattened_norm_gray = [pixel for row in norm_gray1 for pixel in row]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    # Show color image
    axes[0,0].imshow(img_color1)
    axes[0,0].set_title('Original Color Image')
    axes[0,0].axis('off')  # Turn off axis

    # Show grayscale image (normalized)
    axes[0,1].imshow(img_gray1, cmap='gray')
    axes[0,1].set_title('Grayscale Image (Normalized)')
    axes[0,1].axis('off')  # Turn off axis


    #show histogram for unnormalized
    axes[1,0].hist(flattened_gray, bins = 256)
    axes[1,0].set_title('Unnormalized Image')
    axes[1,0].axis('on')  # Turn off axis

    #show histogram for normalized
    axes[1,1].hist(flattened_norm_gray, bins = 256)
    axes[1,1].set_title('Normalized Image')
    axes[1,1].axis('on')  # Turn off axis

    plt.ion() 
    plt.tight_layout()
    plt.show()

    # Wait for a button press
    plt.waitforbuttonpress(0)  # Wait indefinitely until a key press
    plt.close(fig)

  
  #path for Data Loading
img_file_path = "MLA2_DATA/MLA2_DATA/IMAGE"
img_file_dir = os.listdir(img_file_path)
filtered_files_iter = iter(map(lambda x: x * (x != ".DS_Store"), img_file_dir))
img_file_dir = list(filter(None, filtered_files_iter))
#plot 4 random images
for i in random.sample(range(len(img_file_dir)), 4):
    img_lst = os.listdir(f"{img_file_path}/{img_file_dir[i]}")
    for j in random.sample(range(len(img_lst)),1):
        file_path = f"{img_file_path}/{img_file_dir[i]}/{img_lst[j]}"
        img_color, img_gray = preprocess_image(file_path)
        img_gray_norm = normalize_img(img_gray)
        print("Image Data:")
        print(img_gray)
        print("Normalized Image Data")
        print(img_gray_norm)
        plot_image_data(img_color,img_gray)

#1.b

def load_audio_data(file_path):
    label = file_path.split('_')[2]
    sample_rate, audio_data = wavfile.read(file_path)  # Load the audio file
    return label,sample_rate, list(audio_data)  # Convert numpy array to Python list for non-numpy approach

# Function to normalize the audio data to range [-1, 1] using pure Python
def normalize_audio(audio_data):
    audio_data = [float(x) for x in audio_data]
    max_val = max(audio_data)  # Maximum value in the audio data
    min_val = min(audio_data)  # Minimum value in the audio data
    if max_val - min_val == 0:
        return [0] * len(audio_data)    
    # Normalize the audio data between [-1, 1]
    normalized_audio = [2 * (x - min_val) / (max_val - min_val) - 1 for x in audio_data]
    return normalized_audio

# Function to plot unnormalized and normalized audio data with interactive mode
def plot_audio_data(unnormalized_audio, normalized_audio, sample_rate):
    time = [i / sample_rate for i in range(len(unnormalized_audio))]  # Create time axis

    # Create the figure and axes for the two plots (unnormalized and normalized)
    fig1, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Plot the original unnormalized audio
    axes[0].plot(time, unnormalized_audio)
    axes[0].set_title('Unnormalized Audio')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')

    # Plot the normalized audio
    axes[1].plot(time, normalized_audio, color='r')
    axes[1].set_title('Normalized Audio ([-1, 1])')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Amplitude')

    plt.ion()
    plt.tight_layout()
    plt.show()

    # Wait for a button press to close the plot
    plt.waitforbuttonpress(0)  # Wait indefinitely for a key press or button press
    plt.close(fig1)  # Close the current figure once a button is pressed

    
audio_file_path = "MLA2_DATA/MLA2_DATA/AUDIO"
audio_file_dir = os.listdir(audio_file_path)
for i in random.sample(range(len(audio_file_dir)), 4):
    _,sample_rate,aud_data = load_audio_data(f"{audio_file_path}/{audio_file_dir[i]}")
    norm_aud_data = normalize_audio(aud_data)
    print(f"Sample Rate: {sample_rate}")
    print(f"Unnormalized Data: {aud_data}")
    print(f"Normalized Data: {norm_aud_data}")
    plot_audio_data(aud_data,norm_aud_data,sample_rate)



# Q2


#Sequential Sampler

# def load_image_data(image_dir_path):
#     image_class_folders = os.listdir(image_dir_path)
#     image_class_folders_iter = iter(map(lambda folder: folder * (os.path.isdir(os.path.join(image_dir_path, folder))), image_class_folders))
#     image_class_folders = list(filter(None, image_class_folders_iter))  # Get only valid folders
    
#     # Load the data (file paths) for each class (folder)
#     image_data = {}
#     for class_folder in image_class_folders:
#         image_data[class_folder] = list(map(lambda img_file: os.path.join(image_dir_path, class_folder, img_file), os.listdir(os.path.join(image_dir_path, class_folder))))
#     image_data_list = []
#     for i in image_data.values():
#         for j in range(len(i)):
#             _,i[j] = preprocess_image(i[j])
#             image_data_list.append(i[j])
#     # return image_data
#     return image_data_list

# def load_text_data(path):
#     pass
# def load_audio_data(path):
#     pass

# def Sequential_Sampler(data_path):
#     data_type = data_path.split("/")[2]
#     is_image_data = data_type == 'IMAGE'
    
#     # Use map and boolean logic to determine which function to call
#     load_functions = iter([load_image_data, load_text_data])
    
#     # Use map to apply the appropriate function based on the data type
#     selected_data = list(map(lambda func: func(data_path), [load_image_data] * is_image_data + [load_text_data] * (not is_image_data)))
    
#     return selected_data[0]  # Extract the result from the list
# x = Sequential_Sampler(img_dir_path)
# print(len(y))