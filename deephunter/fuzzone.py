from keras import backend as K
import numpy as np
def predict(self, input_data):
    inp = self.model.input
    functor = K.function([inp] + [K.learning_phase()], self.outputs)
    outputs = functor([input_data, 0])
    return outputs
def fetch_function(handler, input_batches, preprocess):
    _, img_batches, _, _, _ = input_batches
    if len(img_batches) == 0:
        return None, None
    preprocessed = preprocess(img_batches)
    layer_outputs = handler.predict(preprocessed)
    # Return the prediction outputs
    return layer_outputs, np.expand_dims(np.argmax(layer_outputs[-1], axis=1),axis=0)

def quantize_fetch_function(handler, input_batches, preprocess, models):
    _, img_batches, _,_,_ = input_batches
    if len(img_batches) == 0:
        return None, None
    preprocessed = preprocess(img_batches)

    layer_outputs = handler.predict(preprocessed)
    results = np.expand_dims(np.argmax(layer_outputs[-1], axis=1),axis=0)
    for m in models:
        r1 =  np.expand_dims(np.argmax(m.predict(preprocessed), axis=1),axis =0)
        results = np.append(results,r1,axis =0)
    # Return the prediction outputs of all models
    return layer_outputs, results



def build_fetch_function(handler, preprocess,models=None):
    def func(input_batches):
        """The fetch function."""
        if models is not None:
            return quantize_fetch_function(
                handler,
                input_batches,
                preprocess,
                models
            )
        else:
            return fetch_function(
                handler,
                input_batches,
                preprocess
            )
    return func


def adptive_coverage_function(handler, cov_num):
    def func(layerouts):
        ptr = np.zeros(cov_num, dtype=np.uint8)
        return handler.update_coverage(layerouts, ptr)

    return func
