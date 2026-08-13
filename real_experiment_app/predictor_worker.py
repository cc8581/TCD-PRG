from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import numpy as np

from .config import AppConfig
from .predictor import TCDPRGPredictor
from .types import FusedScene


def serializable(value):
    if isinstance(value,np.ndarray): return serializable(value.tolist())
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,float) and not np.isfinite(value): return None
    if isinstance(value,dict): return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [serializable(v) for v in value]
    return value


def respond(value): print(json.dumps(serializable(value),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    predictor=TCDPRGPredictor(AppConfig.load(args.config)); last_observation=None
    respond({"ready":True})
    for line in sys.stdin:
        try:
            request=json.loads(line); command=request["command"]
            if command=="predict":
                with np.load(request["scene"],allow_pickle=False) as data:
                    mapping={int(k):int(v) for k,v in zip(data["category_keys"],data["category_values"],strict=True)}
                    scene=FusedScene(data["xyz_m"].copy(),data["rgb"].copy(),
                        data["instance_id"].copy(),data["source_view"].copy(),mapping)
                last_observation=predictor.build_observation(scene,int(request["target"]),
                    int(request["category"]),int(request["region"]),int(request["required"]))
                started=time.perf_counter(); encoded=predictor.policy.encode_observation(last_observation)
                candidates=predictor.policy.generate_candidates(encoded); action=predictor.policy.select_action(candidates)
                if action is None: raise RuntimeError("Model produced no valid action")
                result={"action":action,"inference_seconds":time.perf_counter()-started}
            elif command=="action_executed":
                if last_observation is None: raise RuntimeError("No observation for policy update")
                predictor.policy.update_after_action(request["action"],last_observation); result=True
            elif command=="reset": predictor.reset(); result=True
            elif command=="close": respond({"ok":True,"result":True}); return
            else: raise ValueError(f"Unknown command {command}")
            respond({"ok":True,"result":result})
        except Exception as error:
            respond({"ok":False,"error":f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})


if __name__=="__main__": main()
