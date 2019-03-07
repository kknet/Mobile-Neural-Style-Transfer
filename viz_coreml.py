# This script typically raises some error like
# "Permission denied: '/usr/local/lib/python3.5/dist-packages/coremltools/graph_visualization/model.json'"
# You neeed to create model.json and modify the access right to solve this problem.
import coremltools
import sys

# Load the model
model = coremltools.models.MLModel(sys.argv[1])

# Visualize the model
model.visualize_spec()
