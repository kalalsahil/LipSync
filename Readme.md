# **Lip Sync Model Implementation (Wav2Lip)**

## 📌 **Objective**

This project implements the **Wav2Lip** model to generate lip-synced videos by synchronizing facial movements with speech. Given an **input video** and **audio file**, the model outputs a realistic video where the lips match the speech accurately.

## 🏗 **Project Structure**

```
📁 LIPSYNC
 ├── 📂 env/                 # Conda virtual environment
 ├── 📂 input/               # Folder for input files
 │   ├── input_video.mp4     # Input face video
 │   ├── input.wav           # Input speech/audio
 │   ├── Screenshot.png      # Screenshot (optional)
 │
 ├── 📂 output/              # Folder for generated lip-synced video
 │   ├── result_voice.mp4    # Output video file
 │
 ├── 📂 Wav2Lip/             # Wav2Lip implementation files
 │   ├── 📂 checkpoints/     # Pretrained model files (to be added)
 │   ├── 📂 evaluation/      # Evaluation scripts
 │   ├── 📂 face_detection/  # Face detection module
 │   ├── 📂 filelists/       # File list management
 │   ├── 📂 models/          # Model architecture files
 │   ├── 📂 results/         # Intermediate outputs
 │   ├── 📂 temp/            # Temporary processing files
 │   ├── audio.py            # Audio processing script
 │   ├── inference.py        # Main inference script
 │   ├── preprocess.py       # Preprocessing script
 │   ├── requirements.txt    # Dependency list
 │   ├── wav2lip_train.py    # Training script (optional)
 │   ├── README.md           # Wav2Lip documentation
 │
 ├── Lip_Sync.ipynb          # Jupyter Notebook for implementation
 ├── Readme.md               # Project documentation
```

---

## 🛠 **Installation**

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip
```

### 2️⃣ Install Dependencies

Ensure you have Python 3.7+ installed. Then, install the required libraries:

```bash
pip install -r requirements.txt
```

📌 **Dependencies:**

- librosa==0.7.0
- numpy==1.17.1
- opencv-python==4.1.0.25
- torch==1.1.0
- torchvision==0.3.0
- tqdm==4.45.0
- numba==0.48

### 3️⃣ Download Pretrained Model

The Wav2Lip model requires a pretrained checkpoint file. Download it from the following link:

🔗 [Pretrained Model Download](https://drive.google.com/file/d/1xIMvN1w8bGUT7d9fdWwAJU4_cpGgHpu7/view?usp=drive_link)

After downloading, place the `.pth` file inside the `checkpoints/` directory.

---

## 🎬 **Usage Instructions**

You can also run this project on Google Colab: [Colab Notebook](https://colab.research.google.com/drive/1FAD6Izn_KYaFxZe5xlXNW_sAYrrb7Lzq?usp=sharing)

### **Run via Jupyter Notebook**

1. Open Lip_Sync.ipynb in Jupyter Notebook.
2. Execute all cells step by step.
3. The generated lip-synced video will be saved in the output/ folder.

---

## 📌 **Example Output**

**Input:**

- **Video:** A face speaking with no lip movement.
- **Audio:** Speech file (`input.wav`).

**Output:**

- **Generated video (`result_voice.mp4`)** with synchronized lip movements matching the input audio.

  📹 **Watch the Video**
[Download and Play](https://drive.google.com/file/d/1kh5Xn6jUquemaVINVGciqMJP3xujJpVx/view?usp=drive_link)

---

## 👤 **Contributor**

- **Sahil Kalal**

---
