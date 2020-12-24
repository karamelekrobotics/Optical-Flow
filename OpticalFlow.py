import cv2 
import numpy as np

cap = cv2.VideoCapture(0)

# create old frame
_,frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# LK Params
lk_params = dict(winSize = (15,15),
                maxLevel = 4,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



# mouse function
def SelectPoint(event,x,y,flags,params):
    global point, pointSelected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        pointSelected = True
        old_points = np.array([[x,y]], dtype = np.float32)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",SelectPoint)

pointSelected = False
point = ()
old_points = np.array([[]])

while True:
    _,frame = cap.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    if pointSelected is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)

        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_gray = gray_frame.copy()
        old_points = new_points
        #print(new_points)

        x, y = new_points.ravel()
        cv2.circle(frame, (x,y), 5, (0, 255, 0), -1)

    first_level  = cv2.pyrDown(frame)
    second_level  = cv2.pyrDown(first_level)

    cv2.imshow("Frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
