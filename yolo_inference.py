from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('video/Vinicius_Goal_2024_UCL_Final.mp4', save=True)
print("==============model predicted==============")
for box in results[0].boxes:
    print(box)