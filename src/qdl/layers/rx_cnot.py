import torchquantum as tq
import torchquantum.functional as tqf
import torch

class RxCNOT(tq.QuantumModule):
    """
    Implements a simple quantum layer composed of one Rx
    gate per qubit and pairwise CNOTs to create a final
    entangled state. Based on torchquantum QLSTM gates.
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_wires = cfg.n_wires
        self.rx_gates = torch.nn.ModuleList([
            tq.RX(has_params=True, trainable=True, init_params=cfg.init_params)
            for _ in range(self.n_wires)
        ])

    def forward (self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        for k in range(self.n_wires):
            self.rx_gates[k](self.q_device, wires=k)

        for k in range(self.n_wires):
            if k==self.n_wires-1:
                tqf.cnot(self.q_device, wires=[k, 0])
            else:
                tqf.cnot(self.q_device, wires=[k, k+1])

        return self.q_device