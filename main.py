import time
import argparse

import cv2
import numpy as np
import mediapipe as mp

from Utils import get_face_area
from Eye_Dector_Module import EyeDetector as EyeDet
from Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from Attention_Scorer_Module import AttentionScorer as AttScorer

import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# camera matrix obtained from the camera calibration script, using a 9x6 chessboard
camera_matrix = np.array(
    [[899.12150372, 0., 644.26261492],
     [0., 899.45280671, 372.28009436],
     [0, 0,  1]], dtype="double")

# distortion coefficients obtained from the camera calibration script, using a 9x6 chessboard
dist_coeffs = np.array(
    [[-0.03792548, 0.09233237, 0.00419088, 0.00317323, -0.15804257]], dtype="double")

def _get_landmarks(lms):
    surface = 0
    for lms0 in lms:
        landmarks = [np.array([point.x, point.y, point.z]) \
                        for point in lms0.landmark]

        landmarks = np.array(landmarks)

        landmarks[landmarks[:, 0] < 0., 0] = 0.
        landmarks[landmarks[:, 0] > 1., 0] = 1.
        landmarks[landmarks[:, 1] < 0., 1] = 0.
        landmarks[landmarks[:, 1] > 1., 1] = 1.

        dx = landmarks[:, 0].max() - landmarks[:, 0].min()
        dy = landmarks[:, 1].max() - landmarks[:, 1].min()
        new_surface = dx * dy
        if new_surface > surface:
            biggest_face = landmarks
    
    return biggest_face
i = 0

global detector
detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                            min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5,
                                            refine_landmarks=True)

# instantiation of the eye detector and pose estimator objects
global Eye_det
Eye_det = EyeDet()
global Head_pose
Head_pose = HeadPoseEst()

global t0
t0 = time.perf_counter()

global Scorer
Scorer = AttScorer(t_now=t0)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    frame = frame.to_ndarray(format="bgr24")
    t_now = time.perf_counter()
    fps = i / (t_now - t0)
    i += 1
    if fps == 0:
        fps = 10
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_size = frame.shape[1], frame.shape[0]

    gray = np.expand_dims(cv2.bilateralFilter(gray, 5, 10, 10), axis=2)
    gray = np.concatenate([gray, gray, gray], axis=2)

    lms = detector.process(gray).multi_face_landmarks

    if lms:
        landmarks = _get_landmarks(lms)

        # shows the eye keypoints (can be commented)
        Eye_det.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size)

        # compute the EAR score of the eyes
        ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)

        # compute the PERCLOS score and state of tiredness
        tired, perclos_score = Scorer.get_PERCLOS(t_now, fps, ear)

        # compute the Gaze Score
        gaze = Eye_det.get_Gaze_Score(frame=gray, landmarks=landmarks, frame_size=frame_size)

        # compute the head pose
        frame_det, roll, pitch, yaw = Head_pose.get_pose(frame=frame, landmarks=landmarks, frame_size=frame_size)
            
        # evaluate the scores for EAR, GAZE and HEAD POSE
        asleep, looking_away, distracted = Scorer.eval_scores(t_now=t_now,
                                                                ear_score=ear,
                                                                gaze_score=gaze,
                                                                head_roll=roll,
                                                                head_pitch=pitch,
                                                                head_yaw=yaw)
        
        if frame_det is not None:
            frame = frame_det
            
        if ear is not None:
                cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

        # show the real-time Gaze Score
        if gaze is not None:
            cv2.putText(frame, "Gaze Score:" + str(round(gaze, 3)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

        # show the real-time PERCLOS score
        cv2.putText(frame, "PERCLOS:" + str(round(perclos_score, 3)), (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
        
        if roll is not None:
            cv2.putText(frame, "roll:"+str(roll.round(1)[0]), (450, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)
        if pitch is not None:
            cv2.putText(frame, "pitch:"+str(pitch.round(1)[0]), (450, 70), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)
        if yaw is not None:
            cv2.putText(frame, "yaw:"+str(yaw.round(1)[0]), (450, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)
        

        if tired:
            cv2.putText(frame, "TIRED!", (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        if asleep:
            cv2.putText(frame, "ASLEEP!", (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        if looking_away:
            cv2.putText(frame, "LOOKING AWAY!", (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        if distracted:
            cv2.putText(frame, "DISTRACTED!", (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "FPS:" + str(round(fps)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)

    return av.VideoFrame.from_ndarray(frame, format="bgr24")

def main():

    parser = argparse.ArgumentParser(description='Driver State Detection')

    # selection the camera number, default is 0 (webcam)
    parser.add_argument('-c', '--camera', type=int,
                        default=0, metavar='', help='Camera number, default is 0 (webcam)')

    # TODO: add option for choose if use camera matrix and dist coeffs

    # visualisation parameters
    parser.add_argument('--show_fps', type=bool, default=True,
                        metavar='', help='Show the actual FPS of the capture stream, default is true')
    parser.add_argument('--show_proc_time', type=bool, default=True,
                        metavar='', help='Show the processing time for a single frame, default is true')
    parser.add_argument('--show_eye_proc', type=bool, default=False,
                        metavar='', help='Show the eyes processing, deafult is false')
    parser.add_argument('--show_axis', type=bool, default=True,
                        metavar='', help='Show the head pose axis, default is true')
    parser.add_argument('--verbose', type=bool, default=False,
                        metavar='', help='Prints additional info, default is false')

    # Attention Scorer parameters (EAR, Gaze Score, Pose)
    parser.add_argument('--smooth_factor', type=float, default=0.5,
                        metavar='', help='Sets the smooth factor for the head pose estimation keypoint smoothing, default is 0.5')
    parser.add_argument('--ear_thresh', type=float, default=0.2,
                        metavar='', help='Sets the EAR threshold for the Attention Scorer, default is 0.2')
    parser.add_argument('--ear_time_thresh', type=float, default=2,
                        metavar='', help='Sets the EAR time (seconds) threshold for the Attention Scorer, default is 2 seconds')
    parser.add_argument('--gaze_thresh', type=float, default=0.015,
                        metavar='', help='Sets the Gaze Score threshold for the Attention Scorer, default is 0.2')
    parser.add_argument('--gaze_time_thresh', type=float, default=2, metavar='',
                        help='Sets the Gaze Score time (seconds) threshold for the Attention Scorer, default is 2. seconds')
    parser.add_argument('--pitch_thresh', type=float, default=20,
                        metavar='', help='Sets the PITCH threshold (degrees) for the Attention Scorer, default is 30 degrees')
    parser.add_argument('--yaw_thresh', type=float, default=20,
                        metavar='', help='Sets the YAW threshold (degrees) for the Attention Scorer, default is 20 degrees')
    parser.add_argument('--roll_thresh', type=float, default=20,
                        metavar='', help='Sets the ROLL threshold (degrees) for the Attention Scorer, default is 30 degrees')
    parser.add_argument('--pose_time_thresh', type=float, default=2.5,
                        metavar='', help='Sets the Pose time threshold (seconds) for the Attention Scorer, default is 2.5 seconds')

    args = parser.parse_args()

    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)
        except:
            print(
                "OpenCV optimization could not be set to True, the script may be slower than expected")

    global detector
    detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5,
                                               refine_landmarks=True)

    # instantiation of the eye detector and pose estimator objects
    global Eye_det
    Eye_det = EyeDet(show_processing=args.show_eye_proc)

    global Head_pose
    Head_pose = HeadPoseEst(show_axis=args.show_axis)

    # instantiation of the attention scorer object, with the various thresholds
    # NOTE: set verbose to True for additional printed information about the scores
    global t0
    t0 = time.perf_counter()
    
    global Scorer
    Scorer = AttScorer(t_now=t0, ear_thresh=args.ear_thresh, gaze_time_thresh=args.gaze_time_thresh,
                       roll_thresh=args.roll_thresh, pitch_thresh=args.pitch_thresh,
                       yaw_thresh=args.yaw_thresh, ear_time_thresh=args.ear_time_thresh,
                       gaze_thresh=args.gaze_thresh, pose_time_thresh=args.pose_time_thresh,
                       verbose=args.verbose)
        
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    return


if __name__ == "__main__":
    main()
