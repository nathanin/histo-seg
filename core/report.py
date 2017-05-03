'''
Class 'Report' for tracking statistics concerning the processing
time and speed

Glorified dictionary

'''

import time
import sys
import os
import numpy as np

class Report:

    def __init__(self, project, filename):
        self.project_dir = os.path.join(project, 'Report')
        if not os.path.exists(self.project_dir):
            os.mkdir(self.project_dir)

        self.filename = filename

    def start_filename(self):
        # Check for file existence and record it


