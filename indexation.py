
# Traitement complet d'une vidéo : extraction de frames, détection d'objets, transcription audio, création d'un index lisible.

import os
import cv2
import whisper
import moviepy as mp
from pathlib import Path
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
import os, subprocess

# 1) Determine script folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2) Build path to your bundled ffmpeg.exe folder
FFMPEG_BIN = os.path.join(SCRIPT_DIR, "ffmpeg", "bin")
# 3) Prepend to PATH
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

print("PATH in Python:", os.environ["PATH"])
try:
    subprocess.run(["ffmpeg", "-version"], check=True)
    print("FFmpeg is visible to Python!")
except Exception as e:
    print("FFmpeg NOT visible:", e)



# ------------ Etape 1 : Extraction de keyframes ------------
def extract_keyframes(video_path, output_folder, interval_sec=1):
    os.makedirs(output_folder, exist_ok=True)#make a directory with the output path and ignore errors if the folder already exists
    cap = cv2.VideoCapture(video_path)#decomposition of the video into images (frames)
    fps = cap.get(cv2.CAP_PROP_FPS)#gets the video propreties we used the frame rate of the video fps
    interval = int(fps * interval_sec)#interval_sec is the interval between keyframes that we want to save we multiply it by the fps so we can know how many frames we skip before saving the next keyframe
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()#gets the next frame of the video the read methode returns a boolean and an array of the pixel values for that frame
        if not ret:#if the frame was not read succesfully
            break
        if frame_count % interval == 0:#when the frame count is divisible by the intervale its time to save the frame 
            filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")#generating a filename for the keyframe in the ouput folder with a 4digit padded index
            cv2.imwrite(filename, frame)#saves the current frame int the specified filename
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"{saved_count} keyframes saved to {output_folder}")


# ------------ Etape 2 : Détection d'objets avec YOLOv5 ------------
def detect_objects_yolo(frame_folder):
    model = YOLO("yolov5s.pt")
    results = []

    for img_path in Path(frame_folder).glob("*.jpg"):#looping through all the frame files ending with jpg extension and we used path to convert the path from a string to a path object using pathlib
        results_img = model(str(img_path))#runs the object detection model in the current image  returns a prediction result list
        result = results_img[0]
        df = result.to_df()#convert all the detections into a pandas dataframe and use the xyxy format that gives us the topleft and bottom right corners
        print(df.columns)     

        for _, row in df.iterrows():
            print(type(row['box']), row['box'])
            box = row['box']
            x1 = box['x1']
            y1 = box['y1']
            x2 = box['x2']
            y2 = box['y2']
            results.append({
                "frame":      img_path.name,#the filename where it detected the object
                "label":      row['name'],#the class name of the detected object
                "confidence": float(row['confidence']),#the confidence of the detected object
                "x1":         float(x1),
                "y1":         float(y1),
                "x2":         float(x2),
                "y2":         float(y2)
            })
       

    print(f"{len(results)} objets détectés.")
    return results


# ------------ Etape 3 : Transcription audio avec Whisper ------------
def transcribe_audio(video_path, audio_path="temp_audio.wav"):
    clip = mp.editor.VideoFileClip(video_path)#loading the video with the moviepy.editor 
    clip.audio.write_audiofile(audio_path)#clip.audio grabs the audio track from the video and writes it in the .wav file

    model = whisper.load_model("base")#loading the openai model whipser and the model size the larger the model the more accurate it is 
    result = model.transcribe(audio_path,word_timestamps=True)#prefroms the transcription on the extracted audio and returns a dictinoarry with several keys 
    full_text = result["text"]
    segments = result["segments"]
    print("Transcription terminée.")
    return full_text, segments #we want only the text not the full meta data

def search_transcript_with_timestamps(segments, query):
    matches = []
    for segment in segments:
        for word_info in segment.get("words", []):
            if query.lower() in word_info["word"].lower():
                matches.append({
                    "word": word_info["word"],
                    "time": round(word_info["start"], 2)
                })
    return matches
# ------------ Etape 4 : Création de l’index final (fichier texte) ------------
def create_index_txt(object_data, transcript, output_path="video_index.txt"):#combining the object detection data and the transcribed speecch into a txt file
    cap = cv2.VideoCapture('video.mp4')
    video_props = {
    "Frame Width": cv2.CAP_PROP_FRAME_WIDTH,
    "Frame Height": cv2.CAP_PROP_FRAME_HEIGHT,
    "Frame Count": cv2.CAP_PROP_FRAME_COUNT,
    "FPS": cv2.CAP_PROP_FPS,
    "FourCC": cv2.CAP_PROP_FOURCC,
    "Frame Position": cv2.CAP_PROP_POS_FRAMES,
    "Milliseconds Position": cv2.CAP_PROP_POS_MSEC,
    "Brightness": cv2.CAP_PROP_BRIGHTNESS,
    "Contrast": cv2.CAP_PROP_CONTRAST,
    "Saturation": cv2.CAP_PROP_SATURATION,
    "Hue": cv2.CAP_PROP_HUE,
    "Gain": cv2.CAP_PROP_GAIN,
    "Exposure": cv2.CAP_PROP_EXPOSURE,
    "Convert RGB": cv2.CAP_PROP_CONVERT_RGB
}


    with open(output_path, "w", encoding="utf-8") as f:#opens a file for writing with the file mode w that creates and overwritres the output file
        f.write("=======METADATA===========\n")
        for prop_name, prop_id in video_props.items():
            value = cap.get(prop_id)
            f.write(f"{prop_name}: {value}\n")
        f.write("===== TRANSCRIPTION =====\n")
        f.write(transcript + "\n\n")

        f.write("===== OBJETS DÉTECTÉS =====\n")
        for obj in object_data:
            f.write(
                f"Frame: {obj['frame']}, "
                f"Objet: {obj['label']}, "
                f"Confiance: {obj['confidence']:.2f}, "
                f"Coordonnées: ({obj['x1']:.1f}, {obj['y1']:.1f}) - ({obj['x2']:.1f}, {obj['y2']:.1f})\n"
            )
    print(f"Index sauvegardé dans {output_path}")


# ------------ Pipeline principal ------------
def run_pipeline(video_path):
    frames_folder = "frames"
    extract_keyframes(video_path, frames_folder, interval_sec=1)

    object_data = detect_objects_yolo(frames_folder)

    # Get full transcript + segments with word timestamps
    full_text, segments = transcribe_audio(video_path)

    # Save transcript + object detection to index file
    create_index_txt(object_data, full_text, output_path="video_index.txt")

    # Optional: Let user search words and get timestamps
    while True:
        query = input("🔍 Enter a word to search in the transcript (or 'q' to quit): ").strip()
        if query.lower() == 'q':
            break

        results = search_transcript_with_timestamps(segments, query)

        if results:
            print(f"\n🔎 Found {len(results)} occurrences of '{query}':")
            for r in results:
                print(f"- Word: {r['word']}, Time: {r['time']}s")

            # Save search results
            with open("search_results.txt", "a", encoding="utf-8") as f:
                f.write(f"\nSearch for: '{query}'\n")
                for r in results:
                    f.write(f"Word: {r['word']} | Time: {r['time']}s\n")
        else:
            print(f"❌ No matches found for '{query}'.")


# ------------ Exécution ------------
if __name__ == "__main__":
    video_file = "video.mp4"
    run_pipeline(video_file)
