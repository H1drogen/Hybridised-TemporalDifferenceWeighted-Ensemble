# tdw
Temporal Difference Weighted Ensemble for Reinforcement Learning

## install
### manual
```
$ pip install -r requirements.txt
$ pip install nnabla
```
If you use GPU, see [here](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html).

### Docker
You need to setup nvidia-docker to run this container.
```
$ ./scripts/build.sh
$ ./scripts/up.sh
```

### k8s
```
$ kubectl create -f k8s.yaml
$ kubectl exec -it tdw bash
$ ./scripts/install.sh # on the created Pod
```

## train
```
$ python -m experiments.train --env BreakoutDeterministiv-v4 --seed 0 --gpu --logdir breakout_seed0
```

## evaluate
```
$ python -m experiments.evaluate --env BreakoutDeterministic-v4 --gpu --ensemble tdw_average --decay 0.8 --load model1.h5 model2.h5 ...
```
