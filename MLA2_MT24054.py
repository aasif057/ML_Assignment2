import random
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import json
import os
from PIL import Image
import sys
from scipy.io import wavfile

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
    # Wait for a button press
    plt.waitforbuttonpress(0)  # Wait indefinitely until a key press
    plt.close()
  
  
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
        sys.stdout.flush()



#1.b
def load_audio_file(file_path):
    label = file_path.split('_')[2]
    sample_rate, audio_data = wavfile.read(file_path)  # Load the audio file
    return label,sample_rate, list(audio_data)  
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
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

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
    # Wait for a button press to close the plot
    plt.waitforbuttonpress(0)  # Wait indefinitely for a key press or button press
    plt.close()  # Close the current figure once a button is pressed
    

audio_file_path = "MLA2_DATA/MLA2_DATA/AUDIO"
audio_file_dir = os.listdir(audio_file_path)

# plot 4 random audio files
for i in random.sample(range(len(audio_file_dir)), 4):
    _,sample_rate,aud_data = load_audio_file(f"{audio_file_path}/{audio_file_dir[i]}")
    norm_aud_data = normalize_audio(aud_data)
    print("Sample Rate:")
    print(sample_rate)
    print("Unnormalized Data:")
    print(aud_data)
    print("Normalized Data:")
    print(norm_aud_data)
    plot_audio_data(aud_data,norm_aud_data,sample_rate)
    sys.stdout.flush()

#Q1.c

def open_text_file(file_path):
    """Load JSONL data from a file and return as a list of dictionaries."""
    data_samples = []
    with open(file_path, 'r') as file:
        for line in file:
            data_samples.append(json.loads(line))  # Parse each JSON line and add to list
    return data_samples

def create_dict(data):
    """Create a character-to-number dictionary from the given data."""
    all_text = ''.join([entry['norm'] for entry in data])  # Combine all 'norm' sentences into one string
    unique_chars = sorted(set(all_text))  # Get unique characters
    char_to_num = {char: index for index, char in enumerate(unique_chars)}
    return char_to_num

def tokenize_sentence(sentence, char_to_num):
    """Tokenize a given sentence using the character-to-number dictionary."""
    return [char_to_num[char] for char in sentence]  # List comprehension for tokenization
    
text_file_path = "MLA2_DATA/MLA2_DATA/TEXT/train.jsonl"
data = open_text_file(text_file_path)
data_dict = create_dict(data)
print(data_dict)
for i in random.sample(range(len(data)), 4):
    sentence = data[i]['norm']
    print("Sentence:")
    print(sentence)
    print("Tokenized Sentence:")
    tokenized_sentence = tokenize_sentence(sentence,data_dict)
    print(tokenized_sentence)
    
# Q2

# a    
#Sequential Sampler

def get_img_files_paths(image_dir_path):
    image_class_folders = os.listdir(image_dir_path)  # Get list of class folders
    filtered_files_iter = iter(map(lambda x: x * (x != ".DS_Store"), image_class_folders))
    image_class_folders = list(filter(None, filtered_files_iter))
    image_data = {}  # Dictionary to store class -> image paths

    # Iterate through each class folder and store image paths
    for class_folder in image_class_folders:
        class_folder_path = os.path.join(image_dir_path, class_folder)
        image_files = os.listdir(class_folder_path)  # List of images in the class
        image_data[class_folder] = [os.path.join(class_folder_path, img) for img in image_files]
        
    return image_data

def sort_img_classes(image_data):
    # Sort classes first by the number of samples, and if equal, alphabetically
    sorted_classes = sorted(image_data.items(), key=lambda x: (len(x[1]), x[0]))
    return sorted_classes

def load_image_data(image_dir_path):
    # Load the image data
    image_data = get_img_files_paths(image_dir_path)

    # Sort the classes by number of samples, and alphabetically if tied
    sorted_image_data = sort_img_classes(image_data)

    # Return data in sorted sequential order
    img_data = []
    for class_name, images in sorted_image_data:
        for image_path in images:
            # Use the preprocess_image function to get color and grayscale image arrays
            _,img_gray = preprocess_image(image_path)
            img_data.append({'class':class_name,'data':img_gray})
            sys.stdout.flush()

    return img_data

def load_audio_data(audio_dir_path):
    audio_file_dir = os.listdir(audio_dir_path)
    # Extract class labels and group file names by class
    class_to_files = {}
    
    for file_name in audio_file_dir:
        # Extract label from file name (assuming label is part of the filename)
        label = file_name.split('_')[2]  # Modify as per your filename structure
        class_to_files.setdefault(label, []).append(file_name)

    # Sort classes based on the number of samples and alphabetically if equal
    sorted_classes = sorted(class_to_files.items(), key=lambda x: (len(x[1]), x[0]))
    
    audio_data_list = []

    # Load audio data based on the sorted class labels
    for label, files in sorted_classes:
        for file_name in files:
            # Load audio data for each file
            _,sample_rate, audio_data = load_audio_file(os.path.join(audio_dir_path, file_name))
            audio_data_list.append({"class": label, "sample_rate": sample_rate, "audio_data": audio_data})
            sys.stdout.flush()  # Flush stdout if necessary

    return audio_data_list

def load_text_data(file_path):
    """Return a dataset as a dictionary sorted by class labels."""
    data_samples = open_text_file(file_path)
    # Collect sentences and labels from the dataset
    sentences = [sample['norm'] for sample in data_samples]
    labels = [sample['label'] for sample in data_samples]

    # Create character dictionary
    char_to_num = create_dict(data_samples)

    # Create a dictionary to group sentences by their labels
    grouped_data = {}
    for sentence, label in zip(sentences, labels):
        flag = label == '1'
        label_str = "moral" * (flag) + "immoral" * (not flag)  # Convert labels to strings
        grouped_data.setdefault(label_str, []).append(sentence)

    # Sort classes based on sample counts and alphabetically
    sorted_classes = sorted(grouped_data.items(), key=lambda x: (len(x[1]), x[0]))

    # Prepare final result
    result = []
    for label_str, sentence_list in sorted_classes:
        for sentence in sentence_list:
            tokenized_data = tokenize_sentence(sentence, char_to_num)
            result.append({
                'sentence': sentence,
                'tokenized_data': tokenized_data,
                'label': label_str
            })

    return result

def Sequential_Sampler(data_path):
    data_type = data_path.split("/")[2]
    is_image_data = data_type == 'IMAGE'
    # Use map to apply the appropriate function based on the data type
    selected_data = list(map(lambda func: func(data_path), [load_image_data] * is_image_data + [load_text_data] * (not is_image_data)))
    sys.stdout.flush()
    return selected_data[0]

x = Sequential_Sampler(img_file_path)
print(x)
y = Sequential_Sampler(text_file_path)
print(y)