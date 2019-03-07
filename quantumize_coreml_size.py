import coremltools
from coremltools.models.neural_network.quantization_utils import quantize_weights
import sys

model_in = sys.argv[1]
names = model_in.split(".")
model_out = names[0] + "_quatumized." + names[1]

# if the OS is not macOS or old macOS
# quantize_weights() returns spec rather than model
model =  coremltools.models.MLModel(model_in)
n_bits = 8
mode = "kmeans"
try:
    quatumized_spec = quantize_weights(model, n_bits, mode)
    coremltools.utils.save_spec(quatumized_spec, model_out)
except Exception as err:
    print("macOS version: ", coremltools.models.utils.macos_version())
    print(err)
    quatumized_model = quantize_weights(model, n_bits, mode)
    coremltools.utils.save_spec(quatumized_model.spec, model_out)
