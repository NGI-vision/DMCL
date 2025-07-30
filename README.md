## datasets
- **COD datasets**:
    download the COD datasets set from [here](https://github.com/lartpang/awesome-segmentation-saliency-dataset#camouflaged-object-detection-cod)(CAMO, CHAMELEON, COD10K, NC4K), and put into 'dataset'
    
- **depth datasets**:
    download the depth datasets set from PropNet [here](https://github.com/Zongwei97/PopNet) 
    
- **Scribble datasets**:
    download the depth datasets set from CRNet [here](https://github.com/dddraxxx/Weakly-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations)   
   



### Train
```bash
python train.py
```

### Test

```bash
python test.py
```

### eval

```bash
python MSCAF_COD_evaluation/eval-CAMO.py
python MSCAF_COD_evaluation/eval-CHAMELEON.py
python MSCAF_COD_evaluation/eval-COD10K.py
python MSCAF_COD_evaluation/eval-NC4K.py

```
