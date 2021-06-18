import numpy as np
import numpy.linalg as la

DEFAULT_JOINT_NAMES = [
    'Nose',             # 0
    'Neck',
    'Shoulder right',   # 2
    'Elbow right',      # 3
    'Hand right',       # 4
    'Shoulder left',    # 5
    'Elbow left',
    'Hand left',        # 7
    'Hip right',        # 8
    'Knee right',
    'Foot right',       # 10
    'Hip left',
    'Knee left',        # 12
    'Foot left',        # 13
    'Eye right',
    'Eye left',         # 15
    'Ear right',
    'Ear left'          # 17
]

# GT joints:
# * head (0)                # 0
# * neck (1)
# * left-shoulder (2)
# * left-elbow (3)
# * left-hand (4)
# * right-shoulder (5)      # 5
# * right-elbow (6)
# * right-hand (7)
# * left-hip (8)
# * left-knee (9)
# * left-foot (10)          # 10
# * right-hip (11)
# * right-knee (12)
# * right-foot (13)
DEFAULT_JOINT_TO_GT_JOINT = np.array([
    0,
    1,
    5,   6,  7,  # right arm
    2,   3,  4,  # left arm
    11, 12, 13,  # right leg
    8,   9, 10,  # left leg
    0, 0, 0, 0   # eye - ear
])
DEFAULT_JOINT_TO_GT_JOINT.setflags(write=False)  # read-only


DEFAULT_SYMMETRIC_JOINTS = np.array([
    (2, 5), (3, 6), (4, 7),
    (8, 11), (9, 12), (10, 13),
    (14, 15), (16, 17)
])
DEFAULT_SYMMETRIC_JOINTS.setflags(write=False)  # read-only


DEFAULT_SENSIBLE_LIMB_LENGTH = np.array([
    (0.05, 0.4),  # head - neck
    (0.03, 0.4),  # neck - left shoulder
    (0.03, 0.4),  # neck - right shoulder
    (0.05, 0.5),  # left shoulder - left elbow
    (0.05, 0.5),  # lelf elbow - left wrist
    (0.05, 0.5),  # right shoulder - right elbow
    (0.05, 0.5),  # right elbow-  right wrist           # 5
    (0.3, 0.75),  # neck - pelvis
    (0.03, 0.4),  # pelvis - left hip
    (0.03, 0.4),  # pelvis - right hip
    (0.2, 0.6),  # left hip - left knee
    (0.2, 0.6),  # left knee - left ankle
    (0.2, 0.6),  # right hip - right knee
    (0.2, 0.6),  # right knee - right ankle            # 10
    # (0.2, 0.6),  # knee left - foot left,
    
    # (0.005, 0.2),  # nose - eye right
    # (0.005, 0.2),  # eye right - ear right
    # (0.005, 0.2),  # nose - eye left                    # 15
    # (0.005, 0.2),  # eye left - ear left
    # (0, 0.55),  # shoulder right - ear right
    # (0, 0.55)  # shoulder left - ear left
])
DEFAULT_SENSIBLE_LIMB_LENGTH.setflags(write=False)  # read-only

def test_distance(jid1, jid2, limb):
    lower, higher = DEFAULT_SENSIBLE_LIMB_LENGTH[limb]
    distance = la.norm(jid1 - jid2)
    if lower <= distance <= higher:
        return False
    else:
        return True