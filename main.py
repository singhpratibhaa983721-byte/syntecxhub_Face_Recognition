import cv2

# Face detection model load karein (Ye OpenCV ke saath pehle se aata hai)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Camera open karein
cap = cv2.VideoCapture(0)

print("Project 1: Pratibha's Face Recognition System shuru ho raha hai...")
print("Band karne ke liye keyboard par 'q' dabayein")

while True:
    # Camera se frame read karein
    ret, frame = cap.read()
    if not ret:
        break

    # Image ko grayscale mein badlein (Detection ke liye zaroori hai)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Chehre detect karein
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Har chehre par box aur aapka naam dikhayein
    for (x, y, w, h) in faces:
        # Green box draw karein (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Aapka naam display karein
        cv2.putText(frame, "User: Pratibha", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Result dikhayein
    cv2.imshow('Syntecxhub Project 1 - Face Recognition', frame)
    
    # 'q' dabane par band karein
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sab kuch band karein
cap.release()
cv2.destroyAllWindows() 
