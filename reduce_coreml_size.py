import coremltools
import sys

model_in = sys.argv[1]
names = model_in.split(".")
model_out = names[0] + "_slim." + names[1]

# Load a model, lower its precision, and then save the smaller model.
model_spec = coremltools.utils.load_spec(model_in)
model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
coremltools.utils.save_spec(model_fp16_spec, model_out)
