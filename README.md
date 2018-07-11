# Code for the paper "Episodic Memory Deep Q-Networks" by Zichuan Lin, Tianqi Zhao, Guangwen Yang and Lintao Zhang

If you use this code in your research, please cite the following paper:

    @article{lin2018episodic,
      title={Episodic Memory Deep Q-Networks},
      author={Lin, Zichuan and Zhao, Tianqi and Yang, Guangwen and Zhang, Lintao},
      journal={arXiv preprint arXiv:1805.07603},
      year={2018}
    }

<img src="data/emdqn.PNG" width=50%  />

## Install
You can install it by typing:
```bash
cd emdqn;
pip install -e .
```

## Run emdqn
```bash
cd emdqn/baselines/deepq/experiments/atari;
python train.py --emdqn
```

