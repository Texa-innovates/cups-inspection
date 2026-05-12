import cv2

def list_connected_cameras(max_index=10):
    print("🔍 Checking connected cameras...")
    available_cameras = []

    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            print(f"✅ Camera found at index {index}")
            available_cameras.append(index)
            cap.release()
        else:
            print(f"❌ No camera at index {index}")
    
    if not available_cameras:
        print("🚫 No cameras detected.")
    return available_cameras

if __name__ == "__main__":
    cameras = list_connected_cameras() 