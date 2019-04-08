"""Pretrained Contextual Embeddings
"""

from allennlp.modules.elmo import Elmo, batch_to_ids

def elmo_url(name: str) -> str:
    """ Build the ELMo url based on the default settings.

    name: suffix at the end of the url, for example "options.json"
    """
    base = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo"
    return f"{base}/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_{name}"


def elmo():
    """Build an ELMo Language Model
    """
    options_file = elmo_url("options.json")
    weight_file = elmo_url("weights.hdf5")
    return Elmo(options_file, weight_file, 2, dropout=0)

print(elmo())
