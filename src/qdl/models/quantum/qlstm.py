import torchquantum as tq
import torchquantum.functional as tqf
import torch

class QLSTM(torch.nn.Module):
    class QLayer_forget(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.encoder = tq.GeneralEncoder(
                [
                    {'input_idx': [0], 'func': 'rx', 'wires': [0]},
                    {'input_idx': [1], 'func': 'rx', 'wires': [1]},
                    {'input_idx': [2], 'func': 'rx', 'wires': [2]},
                    {'input_idx': [3], 'func': 'rx', 'wires': [3]},
                ]
            )
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.rx1 = tq.RX(has_params=True, trainable=True)
            self.rx2 = tq.RX(has_params=True, trainable=True)
            self.rx3 = tq.RX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward (self, x, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.encoder(self.q_device, x)
            self.rx0(self.q_device, wires=0)
            self.rx1(self.q_device, wires=1)
            self.rx2(self.q_device, wires=2)
            self.rx3(self.q_device, wires=3)
            for k in range(self.n_wires):
                if k==self.n_wires-1:
                    tqf.cnot(self.q_device, wires=[k, 0])
                else:
                    tqf.cnot(self.q_device, wires=[k, k+1])
            return(self.measure(self.q_device))
    class QLayer_input(tq.QuantumModule):
        def __init__(self):
            super().__init__()    
            self.n_wires = 4
            self.encoder = tq.GeneralEncoder(
                [   
                    {'input_idx': [0], 'func': 'rx', 'wires': [0]},
                    {'input_idx': [1], 'func': 'rx', 'wires': [1]},
                    {'input_idx': [2], 'func': 'rx', 'wires': [2]},
                    {'input_idx': [3], 'func': 'rx', 'wires': [3]}, # TODO: Check this
                ]
            )
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.rx1 = tq.RX(has_params=True, trainable=True)
            self.rx2 = tq.RX(has_params=True, trainable=True)
            self.rx3 = tq.RX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward (self, x, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.encoder(self.q_device, x)
            self.rx0(self.q_device, wires=0)
            self.rx1(self.q_device, wires=1)
            self.rx2(self.q_device, wires=2)
            self.rx3(self.q_device, wires=3)
            for k in range(self.n_wires):
                if k==self.n_wires-1:
                    tqf.cnot(self.q_device, wires=[k, 0]) 
                else:
                    tqf.cnot(self.q_device, wires=[k, k+1])
            return(self.measure(self.q_device))
    class QLayer_update(tq.QuantumModule):
        def __init__(self):
            super().__init__()    
            self.n_wires = 4
            self.encoder = tq.GeneralEncoder(
                [   
                    {'input_idx': [0], 'func': 'rx', 'wires': [0]},
                    {'input_idx': [1], 'func': 'rx', 'wires': [1]},
                    {'input_idx': [2], 'func': 'rx', 'wires': [2]},
                    {'input_idx': [3], 'func': 'rx', 'wires': [3]},
                ]
            )
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.rx1 = tq.RX(has_params=True, trainable=True)
            self.rx2 = tq.RX(has_params=True, trainable=True)
            self.rx3 = tq.RX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward (self, x, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.encoder(self.q_device, x)
            self.rx0(self.q_device, wires=0)
            self.rx1(self.q_device, wires=1)
            self.rx2(self.q_device, wires=2)
            self.rx3(self.q_device, wires=3)
            for k in range(self.n_wires):
                if k==self.n_wires-1:
                    tqf.cnot(self.q_device, wires=[k, 0]) 
                else:
                    tqf.cnot(self.q_device, wires=[k, k+1])
            return(self.measure(self.q_device))
    class QLayer_output(tq.QuantumModule):
        def __init__(self):
            super().__init__()    
            self.n_wires = 4
            self.encoder = tq.GeneralEncoder(
                [   
                    {'input_idx': [0], 'func': 'rx', 'wires': [0]},
                    {'input_idx': [1], 'func': 'rx', 'wires': [1]},
                    {'input_idx': [2], 'func': 'rx', 'wires': [2]},
                    {'input_idx': [3], 'func': 'rx', 'wires': [3]},
                ]
            )
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.rx1 = tq.RX(has_params=True, trainable=True)
            self.rx2 = tq.RX(has_params=True, trainable=True)
            self.rx3 = tq.RX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward (self, x, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.encoder(self.q_device, x)
            self.rx0(self.q_device, wires=0)
            self.rx1(self.q_device, wires=1)
            self.rx2(self.q_device, wires=2)
            self.rx3(self.q_device, wires=3)
            for k in range(self.n_wires):
                if k==self.n_wires-1:
                    tqf.cnot(self.q_device, wires=[k, 0]) 
                else:
                    tqf.cnot(self.q_device, wires=[k, k+1])
            return(self.measure(self.q_device))
    def __init__(self, 
                input_size, 
                hidden_size, 
                batch_first=True,
                backend="default.qubit"
    ):
        super().__init__()
        
        self.n_qubits = 4
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.batch_first = batch_first


        self.clayer_in = torch.nn.Linear(self.concat_size, self.n_qubits)
        self.VQC = torch.nn.ModuleDict({
            'forget_gate': self.QLayer_forget(),
            'input_gate': self.QLayer_input(),
            'update_gate': self.QLayer_update(),
            'output_gate': self.QLayer_output()
        })
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        qdev = tq.QuantumDevice(self.n_qubits, bsz=x.shape[0], device=x.device)
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            x_t = x[:, t, :]

            v_t = torch.cat((h_t, x_t), dim=1)

            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.VQC['forget_gate'](y_t, qdev)))  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.VQC['input_gate'](y_t, qdev)))  # input block
            g_t = torch.tanh(self.clayer_out(self.VQC['update_gate'](y_t, qdev)))  # update block
            o_t = torch.sigmoid(self.clayer_out(self.VQC['output_gate'](y_t, qdev))) # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
            

