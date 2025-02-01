#Official Autonomous Agent File

from leaderboard.autoagents.autonomous_agent.AutonomousAgent import AutonomousAgent # type: ignore

def get_entry_point():
    return 'AutoAgent'


class AutoAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        self.sift_feature_detector = cv.SIFT_create()

        self.target_linear_velocity = 0.4
        self.target_angular_velocity = 0
