import json
from pathlib import Path

MODEL_PATH = Path("public/nn/model.json")

def main():
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))

    topo = data.get("modelTopology", {})
    model_cfg = topo.get("model_config", {}).get("config", {})
    layers = model_cfg.get("layers", [])

    for layer in layers:
        cfg = layer.get("config", {})

        # 1) Fix InputLayer shape key: batch_shape -> batch_input_shape
        if layer.get("class_name") == "InputLayer":
            if "batch_shape" in cfg and "batch_input_shape" not in cfg:
                cfg["batch_input_shape"] = cfg.pop("batch_shape")

        # 2) Fix inbound_nodes format (bug from tfjs converter)
        inbound = layer.get("inbound_nodes")
        if isinstance(inbound, list) and inbound and not isinstance(inbound[0], list):
            new_inbound = []
            for node in inbound:
                args = node.get("args", [])
                node_arr = []
                for arg in args:
                    conf = arg.get("config", {})
                    hist = conf.get("keras_history")
                    # keras_history is like ["prev_layer_name", 0, 0]
                    if isinstance(hist, list) and len(hist) == 3:
                        node_arr.append(hist)
                if node_arr:
                    new_inbound.append(node_arr)
            layer["inbound_nodes"] = new_inbound

    MODEL_PATH.write_text(json.dumps(data), encoding="utf-8")
    print("Patched:", MODEL_PATH)

if __name__ == "__main__":
    main()
