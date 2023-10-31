import numpy as np
def read_rig_anim_from_json(anim_file):
    """
    Reads a rig animation from a json file

    Parameters
    ----------
    anim_file : str
        Path to the json file containing the rig animation
    """
    import json
    with open(anim_file) as f:
        rig = json.load(f)


    P = np.array(rig["P"])

    return P