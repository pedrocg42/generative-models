import os

# Path to the folder where FFHQ 128x128 is located (downloaded from https://drive.google.com/drive/folders/1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv)
ffhq_128_image_folder_path = "G:/Pedro/Images/FFHQ/128_128"

# Path to where save the models
model_path = "results/models"
os.makedirs(model_path, exist_ok=True)

# Path to where save the logs
logs_path = "results/logs"
os.makedirs(logs_path, exist_ok=True)
