import coremltools
from coremltools.models.neural_network.quantization_utils import quantize_weights
import sys

model_in = sys.argv[1]
names = model_in.split(".")
model_out = names[0] + "_quatumized." + names[1]

# quatumized the model
# if the OS is not mac OS quantize_weights returns spec
model =  coremltools.models.MLModel(model_in)
n_bits = 8
mode = "kmeans"
if coremltools.models.utils.macos_version() == ():
    quatumized_spec = quantize_weights(model, n_bits, mode)
    coremltools.utils.save_spec(quatumized_spec, model_out)
