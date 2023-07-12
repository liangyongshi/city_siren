#### evaluation.py

You can evaluate our model by using

```
python evaluation.py --eval <evaluation mode> --pemode <encoding mode> --checkpoint_path <saved model> --ply_path <saved point cloud>
```

For example, you could use the following command to evaluate meanIoU

```
python evaluation.py --eval iou --pemode fourier --checkpoint_path /home/user/pyProject/experiment_results/fourier_5/g.pth
```

For example, you could use the following command to evaluate ChamferDistance

```
python evaluation.py --eval cd --pemode xyz --ply_path logs/910fourier.ply
```

