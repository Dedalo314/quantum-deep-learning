# Quantum Deep Learning (QDL)

In this repository the code for the TFM titled "DESIGN AND IMPLEMENTATION OF QUANTUM COMPUTING SYSTEMS FOR MACHINE LEARNING APPLICATIONS" is stored. The code contains implementations of deep learning models with quantum components, such as the Quantum LSTM (QLSTM) or Quanvolutional Neural Networks. To learn how [torchquantum](https://github.com/mit-han-lab/torchquantum) can be integrated with [peft](https://github.com/huggingface/peft) to perform Quantum [QLoRA](https://arxiv.org/abs/2305.14314) and easily finetune LLMs, see the last commits of https://github.com/Dedalo314/peft.

To run a training just build and start a docker container (CUDA needs to be installed on the host):

```bash
$ docker build . -t qdl 
$ docker run --rm --runtime=nvidia --gpus all -v /home/user/quantum-deep-learning/checkpoints:/QDL/checkpoints qdl training_classifier.py model=cnn-qlstm-classifier data=mnist-classification ++trainer.default_root_dir=/QDL/checkpoints/models-quanvolutional-lstm-mnist-classification ++model.optim.lr=1e-3
```
