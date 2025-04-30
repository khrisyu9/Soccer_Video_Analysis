from roboflow import Roboflow
rf = Roboflow(api_key="5RCIDUQ8OWLrDZdMxLc7")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov5")

dataset.location
import subprocess

command = [
    "yolo",
    "task=detect",
    "mode=train",
    "model=yolov5x.pt",
    f"data={dataset.location}/data.yaml",
    "epochs=100",
    "imgsz=640"
]

print("#######start training########")
subprocess.run(command)
