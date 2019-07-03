import cv2
import numpy as np
import asciimatcher
import sobel


def track(input, display=True, output=None):
    """
    apply particle filter to track something in a video
    :param input: the path string of the input video
    """
    out = None
    output_frames = {}
    cap = cv2.VideoCapture(input)
    i = 0
    if cap.isOpened():
        ret, frame = cap.read()
        i += 1
        gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if output is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output/' + output + '.avi',
                                  fourcc, 20.0, (gr.shape[1], gr.shape[0]))
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            # edited_frame = cv2.Canny(frame[:, :, 2], 50, 100)
            edited_frame = asciimatcher.transform_image(frame)
            if display:
                cv2.imshow('ascii', edited_frame)
            if out is not None:
                out.write(frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            i += 1
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    return output_frames


if __name__ == '__main__':
    track("/dev/video0")
