import numpy as np
def read_rig_anim_from_json(anim_file):
    import json
    with open(anim_file) as f:
        rig = json.load(f)


    P = np.array(rig["P"])

    return P