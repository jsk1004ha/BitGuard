import numpy as np

from bitguard_bnn.rbc_model import DetectionFirstBNN
from bitguard_bnn.rbc_edge import export_rbc_edge, PackedRBCRuntime


def test_single_attack_type_never_emits_index_one(tmp_path):
    model = DetectionFirstBNN(4, [4, 2], attack_classes=1).eval()
    model.type_head.bias.data.fill_(1.)
    path = tmp_path / "edge.npz"
    export_rbc_edge(model, path, type_labels=["flood_like"])
    output = PackedRBCRuntime.load(path).predict(np.ones((3, 4), dtype=np.float32))
    assert output.attack_types.tolist() == [0, 0, 0]


def test_constant_output_bias_does_not_overflow(tmp_path):
    model = DetectionFirstBNN(4, [4, 2]).eval()
    model.detector.output.weight.data.zero_()
    model.detector.output.bias.data.fill_(1.)
    path = tmp_path / "constant.npz"
    export_rbc_edge(model, path, detection_threshold=0)
    output = PackedRBCRuntime.load(path).predict(np.ones((2, 4), dtype=np.float32))
    assert output.is_attack.all()
    assert (output.detection_scores > 0).all()
